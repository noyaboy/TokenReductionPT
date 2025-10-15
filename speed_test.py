# https://github.com/zhijian-liu/torchprofile
import time
import resource
import argparse
from contextlib import suppress
from statistics import mean

import pynvml
import wandb
import torch
from timm.models import create_model
import torch.backends.cudnn as cudnn

import models
import utils
from datasets import build_dataset
from train import get_args_parser, adjust_config, count_params, set_seed, set_run_name, build_model


def setup_environment(args):
    set_seed(args.seed)
    cudnn.benchmark = True
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # adjust input sizes
    args.resize_size = args.resize_size if args.resize_size else int(args.input_size / 0.875)
    args.test_input_size = args.test_input_size if args.test_input_size else args.input_size
    args.test_resize_size = args.test_resize_size if args.test_resize_size else int(args.test_input_size / 0.875)

    dataset_val, args.num_classes = build_dataset(is_train=False, args=args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    set_run_name(args)

    if not args.debugging:
        wandb.init(project=args.wandb_project, entity=args.wandb_group, settings=wandb.Settings(start_method="fork"), config=args)
        wandb.run.name = args.run_name

    model = build_model(args)

    model.to(args.device)

    model.eval()

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)

    return model, data_loader_val


@torch.no_grad()
def measure_tp(model, loader, device='cuda', dummy_loader=False,
               multiple=1, warmup_iters=100, amp=True):
    # power in watts
    if torch.cuda.is_available():
        # miliwatts: max or instantaneous?
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        power_watts = [pynvml.nvmlDeviceGetPowerUsage(handle) / 1000]
    else:
        # requires special permissions
        # import pyRAPL

        # pyRAPL.setup()

        # meter = pyRAPL.Measurement('test')
        # meter.begin()

        # # Your CPU code here
        # x = sum(i*i for i in range(10**6))

        # meter.end()
        # print(meter.result.pkg)   # Energy in microjoules
        print('CPU currently does not support power measurement')
        power_watts = [0]


    if torch.cuda.is_available() and amp:
        amp_autocast = torch.cuda.amp.autocast
    else:
        amp_autocast = suppress


    images, targets = next(iter(loader))
    images = images.to(device, non_blocking=True)

    if warmup_iters:
        print(f'Warm-up for {warmup_iters} iterations')
        for _ in range(warmup_iters):
            with amp_autocast():
                model(images)

            if torch.cuda.is_available():
                power_watts.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000)


    if dummy_loader:
        loader = [(images, targets) for _ in range(len(loader))]


    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()


    num_images = 0
    num_batches = 0

    for m in range(multiple):
        for i, (images, _) in enumerate(loader):
            if not dummy_loader:
                images = images.to(device, non_blocking=True)

            with amp_autocast():
                model(images)

            num_images += images.size(0)
            num_batches += 1

            if i % 100 == 0:
                print(f'{m} / {multiple}: {i} / {len(loader)}')


    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_total = time.time() - start


    throughput = round((num_images / time_total), 4)
    latency_ms = round((time_total / num_batches) * 1000.0, 2)  # ms/img
    time_total_mins = round((time_total / 60), 4)
    power_watts = round(mean(power_watts), 2)
    return throughput, latency_ms, time_total_mins, power_watts


def main():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--test_multiple', type=int, default=5,
                        help='test multiple loops (to reduce model loading time influence)')
    parser.add_argument('--dummy_loader', action='store_true',
                        help='use torch.rand() instead of loader (max speed with no loader overhead)')
    parser.add_argument('--warmup_iters', type=int, default=100)
    args = parser.parse_args()
    adjust_config(args)


    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    else:
        baseline_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


    model, test_loader = setup_environment(args)


    tp, latency_ms, time_total, power_watts = measure_tp(
        model, test_loader, args.device, args.dummy_loader,
        args.test_multiple, args.warmup_iters, args.use_amp,
    )

    # gigaflops
    profiler = 'thop' if 'ats' in args.model else 'torchprofile'
    flops = utils.count_flops(model, args.input_size, args.device, profiler=profiler)
    flops = round(flops / 1e9, 4)

    # megabytes of memory
    if torch.cuda.is_available():
        max_memory = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 4)
    else:
        max_memory = round((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - baseline_mem) / (1024 ** 2), 4)

    # params in millions
    no_params = count_params(model)
    no_params = round(no_params / (1e6), 4)

    no_params_trainable = count_params(model, trainable=True)
    no_params_trainable = round(no_params_trainable / (1e6), 4)


    if not args.debugging:
        wandb.run.summary['throughput'] = tp
        wandb.run.summary['latency'] = latency_ms
        wandb.run.summary['time_total'] = time_total
        wandb.run.summary['flops'] = flops
        wandb.run.summary['max_memory'] = max_memory
        wandb.run.summary['no_params'] = no_params
        wandb.run.summary['no_params_trainable'] = no_params_trainable
        wandb.run.summary['power'] = power_watts
        wandb.finish()

    print('run_name,tp,latency_ms,time_total,flops,max_memory,no_params,no_params_trainable,power')
    line = f'{args.run_name},{tp},{latency_ms},{time_total},{flops},{max_memory},{no_params},{no_params_trainable},{power_watts}'
    print(line)
    return 0


if __name__ == "__main__":
    main()
