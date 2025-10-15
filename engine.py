import os
import math
import torch

from typing import Iterable, Optional
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from timm.utils.clip_grad import dispatch_clip_grad

from contextlib import suppress

import utils
import wandb

def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, total_epochs: int,
                    loss_scaler, lr_scheduler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    transform_fn=None,  amp_autocast = suppress,
                    set_training_mode: bool = True,  grad_accum_steps: int = 1, 
                    num_steps_epoch: int = 1000, print_freq: int = 100, multi_label: bool = False,
                    plot_gradients: bool = False, gradient_list=[],
                    save_images_wandb: bool = False, output_dir=None, debugging: bool = False):

    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    for lr_idx in range(len(optimizer.param_groups)):
        metric_logger.add_meter(f'lr-{lr_idx}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, total_epochs)

    # this attribute is added by timm on one optimizer (adahessian)
    is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    total_step = epoch * num_steps_epoch

    epoch_step = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        for param_group in optimizer.param_groups:
            if epoch < param_group['fix_step']:
                param_group["lr"] = 0
        
        # Setup grad accumulation check
        epoch_step += 1
        opt_step = (epoch_step % grad_accum_steps == 0) or (epoch_step == len(data_loader))

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # gpu transform
        if transform_fn is not None:
            samples = transform_fn(samples)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with amp_autocast():
            output = model(samples)
            if multi_label:
                if targets.dim() == 3:
                    targets = targets.max(dim=1)[0]
                if isinstance(output, (tuple, list)):
                    output = list(output)                    
                    output[0] = output[0].float()
                else:
                    output = output.float()

            loss = criterion(samples, output, targets, model)

        loss_value = loss.item() / grad_accum_steps
        loss = loss / grad_accum_steps

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=max_norm, clip_mode="norm",
                parameters=model.parameters(),
                create_graph=is_second_order,
                opt_step = opt_step)
        else:
            loss.backward(create_graph=is_second_order)

            if utils.is_main_process() and plot_gradients:
                gradients = utils.get_gradients(model.named_parameters())
                gradient_list.append(gradients)
                if not debugging:
                    wandb.log(gradients)

            if max_norm is not None and max_norm > 0.:
                dispatch_clip_grad(
                    model.parameters(),
                    value=max_norm, mode="norm")
            if opt_step:
                optimizer.step()

        if opt_step:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

            total_step += 1

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        lr_dict = {}
        for lr_idx in range(len(optimizer.param_groups)):
            lr_dict[f"lr-{lr_idx}"] = optimizer.param_groups[lr_idx]["lr"]
        metric_logger.update(**lr_dict)

        if mixup_fn is not None or len(targets.shape) == 2:
            acc1, acc5 = torch.torch.Tensor([0]), torch.Tensor([0])
        elif isinstance(output, (tuple, list)):
            acc1, acc5 = accuracy(output[0], targets, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))

        batch_size = samples.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        if utils.is_main_process():
            if (total_step % print_freq == 0 or epoch_step == len(data_loader)):
                log_dict = {"train_loss_step": loss_value,
                            **{f"lr-{lr_idx}": optimizer.param_groups[lr_idx]["lr"] for lr_idx in range(len(optimizer.param_groups))},
                            "epoch": epoch+1}

                first_or_last = (total_step == num_steps_epoch or total_step == total_epochs * num_steps_epoch)
                if first_or_last:
                    augs = utils.save_images(samples, output_dir, total_step)

                if isinstance(output, tuple) and len(output) in (3, 4) and first_or_last:
                    augs = utils.save_images(output, output_dir, total_step)

                if not debugging:
                    wandb.log(log_dict, step=total_step)
                    if save_images_wandb and isinstance(output, tuple) and len(output) in (3, 4) and first_or_last:
                        wandb.log({'augs': wandb.Image(augs)})

                else:
                    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, total_step
                
        if lr_scheduler is not None and opt_step:
            lr_scheduler.step_update(num_updates=total_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, total_step


    
@torch.no_grad()
def evaluate_multiclass(data_loader, model, device, amp_autocast = suppress):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            output = model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]

        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate_multilabel(data_loader, model, device, amp_autocast = suppress):
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    Sig = torch.nn.Sigmoid()
    preds_regular = []
    targets = []

    #coco_style = target.dim() == 3

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        target = target
        if target.dim() == 3:
            target = target.max(dim=1)[0].float()
        else:
            target = target.float()
        # compute output
        with amp_autocast():
            output = model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]
        
        output = output.float()
        loss = criterion(output, target)
        output_regular = Sig(output).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        targets.append(target.cpu().detach())

        metric_logger.update(loss=loss.item())

    mAP_score = utils.mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    metric_logger.meters['acc1'].update(mAP_score, n=1)
    metric_logger.meters['acc5'].update(mAP_score, n=1)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
