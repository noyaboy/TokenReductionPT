import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile

import torch
from torchvision import datasets
import torch.utils.data as data
import torchvision.transforms.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

import utils

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except:
    from PIL.Image import BICUBIC as BICUBIC

try:
    from torchvision.transforms import v2 as transforms
except:
    from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True

MEANS = {
    '05': (0.5, 0.5, 0.5),
    'imagenet': (0.485, 0.456, 0.406),
    'cifar10': (0.4914, 0.4822, 0.4465)
}

STDS = {
    '05': (0.5, 0.5, 0.5),
    'imagenet': (0.229, 0.224, 0.225),
    'cifar10': (0.2470, 0.2435, 0.2616)
}


class ResizeAndPad:
    def __init__(self, image_size, padding_value=0):
        self.image_size = image_size
        self.padding_value = padding_value

    def __call__(self, img):
        # Resize the image so that the long side is image_size
        w, h = img.size
        if w > h:
            new_w = self.image_size
            new_h = int(self.image_size * h / w)
        else:
            new_h = self.image_size
            new_w = int(self.image_size * w / h)
        
        img = F.resize(img, (new_h, new_w))
        
        # Calculate padding
        pad_h = self.image_size - new_h
        pad_w = self.image_size - new_w
        
        # Padding on all sides
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        
        img = F.pad(img, padding, fill=self.padding_value)
        
        return img


class RandomResizePadCrop:
    def __init__(self, image_size, resize_size, padding_value=0, random_pad=True):
        self.image_size = image_size
        self.resize_size = resize_size
        self.padding_value = padding_value
        self.random_pad = random_pad
        self.crop = transforms.RandomCrop((image_size, image_size))

    def __call__(self, img):
        # Resize the image so that the long side is image_size
        w, h = img.size
        rand_size = random.randint(self.image_size, self.resize_size)
        # print(img.size, self.image_size, self.resize_size, rand_size)

        if w > h:
            new_w = rand_size
            new_h = int(rand_size * h / w)
        else:
            new_h = rand_size
            new_w = int(rand_size * w / h)
        
        img = F.resize(img, (new_h, new_w), interpolation=BICUBIC)
        # print(img.size)

        # Calculate padding
        pad_w = rand_size - new_w
        pad_h = rand_size - new_h

        if self.random_pad:
            rand_w = random.randint(0, pad_w)
            pad_w = (rand_w, pad_w - rand_w)
            rand_h = random.randint(0, pad_h)
            pad_h = (rand_h, pad_h - rand_h)
        else:
            pad_w = (pad_w // 2, pad_w - pad_w // 2)
            pad_h = (pad_h // 2, pad_h - pad_h // 2)

        # Padding on all sides
        padding = (pad_w[0], pad_h[0], pad_w[1], pad_h[1])
        # print(padding)

        pad_value = random.randint(0, 255)
        img = F.pad(img, padding, fill=pad_value)

        # square crop
        img = self.crop(img)

        return img


class DatasetImgTarget(data.Dataset):
    def __init__(self, args, split, transform=None):
        self.args = args
        self.root = os.path.abspath(args.dataset_root_path)
        self.transform = transform
        self.dataset_name = args.dataset_name


        if self.args.transform_gpu:
            t = []

            if split == 'train' and args.short_side_resize_random_crop:
                input_size = args.input_size
                t.append(transforms.RandomCrop((input_size, input_size)))

            t.append(transforms.ToImage())

            mean = MEANS['imagenet']
            std = STDS['imagenet']
            if args.custom_mean_std:
                mean = MEANS[args.dataset_name] if args.dataset_name in MEANS.keys() else MEANS['05']
                std = STDS[args.dataset_name] if args.dataset_name in STDS.keys() else STDS['05']

            if split != 'train':
                t.append(transforms.ToDtype(torch.float32, scale=True))
                t.append(transforms.Normalize(mean=mean, std=std, inplace=True))

            self.transform_cpu = transforms.Compose(t)
            print('CPU transform:\n:', self.transform_cpu)


        if split == 'train':
            if args.train_trainval:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_trainval
            else:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_train
        elif split == 'val':
            if args.train_trainval:
                self.images_folder = args.folder_test
                self.df_file_name = args.df_test
            else:
                self.images_folder = args.folder_val
                self.df_file_name = args.df_val
        else:
            self.images_folder = args.folder_test
            self.df_file_name = args.df_test

        assert os.path.isfile(os.path.join(self.root, self.df_file_name)), \
            f'{os.path.join(self.root, self.df_file_name)} is not a file.'

        self.df = pd.read_csv(os.path.join(self.root, self.df_file_name), sep=',')
        self.targets = self.df['class_id'].to_numpy()
        self.data = self.df['dir'].to_numpy()

        self.num_classes = len(np.unique(self.targets))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir, target = self.data[idx], self.targets[idx]
        full_img_dir = os.path.join(self.root, self.images_folder, img_dir)
        img = Image.open(full_img_dir)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
        if hasattr(self, 'transform_cpu'):
            img = self.transform_cpu(img)

        return img, target

    def __len__(self):
        return len(self.data)


class DatasetImgTargetInMemory(DatasetImgTarget):
    def __init__(self, args, split, transform=None):
        super().__init__(args, split, transform)
        resize_size = args.resize_size if split == 'train' else args.test_resize_size

        self.images = [None] * len(self.data)

        def load_image(i, img_dir):
            full_img_dir = os.path.join(self.root, self.images_folder, img_dir)
            img = Image.open(full_img_dir)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            if self.args.transform_gpu:
                img = transform(img)
            else:
                img = resize_keep_aspect(img, resize_size, args.short_side_resize_random_crop)

            self.images[i] = img.copy()
            img.close()

        max_workers = 4 if (not args.num_workers or args.num_workers < 4) else args.num_workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(load_image, i, img_dir) for i, img_dir in enumerate(self.data)]
            for f in futures:
                f.result()  # Wait for all threads to complete

        print(f'Loaded all {split} images on memory: {len(self.images)}')

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, target = self.images[idx], self.targets[idx]

        if self.transform and not self.args.transform_gpu:
            img = self.transform(img)
        if hasattr(self, 'transform_cpu'):
            img = self.transform_cpu(img)

        return img, target


class DatasetImgTargetInMemorySharded(DatasetImgTarget):
    def __init__(self, args, split, transform=None):
        super().__init__(args, split, transform)
        resize_size = args.resize_size if split == 'train' else args.test_resize_size

        # Get DDP rank info
        rank = utils.get_rank()
        world_size = utils.get_world_size()

        self.targets = self.targets[rank::world_size]
        self.data = self.data[rank::world_size]

        self.images = [None] * len(self.data)

        def load_image(i, img_dir):
        # for i, img_dir in enumerate(self.data):
            full_img_dir = os.path.join(self.root, self.images_folder, img_dir)
            img = Image.open(full_img_dir)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            if self.args.transform_gpu:
                img = transform(img)
            else:
                img = resize_keep_aspect(img, resize_size, args.short_side_resize_random_crop)

            self.images[i] = img.copy()
            img.close()

        max_workers = 4 if (not args.num_workers or args.num_workers < 4) else args.num_workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(load_image, i, img_dir) for i, img_dir in enumerate(self.data)]
            for f in futures:
                f.result()  # Wait for all threads to complete

        print(f'Loaded rank {rank} for split {split} images on memory: {len(self.images)}')

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, target = self.images[idx], self.targets[idx]

        if self.transform and not self.args.transform_gpu:
            img = self.transform(img)
        if hasattr(self, 'transform_cpu'):
            img = self.transform_cpu(img)

        return img, target

def resize_keep_aspect(img, resize_size, keep_aspect_ratio=True):
    if not keep_aspect_ratio:
        return img.resize((resize_size, resize_size))

    width, height = img.size

    if width > height:
        aspect = width / height
        new_height = resize_size
        new_width = int(resize_size * aspect)
    else:
        aspect = height / width
        new_width = resize_size
        new_height = int(resize_size * aspect)

    img = img.resize((new_width, new_height))
    return img


def build_dataset(is_train, args):
    if args.transform_timm:
        transform = build_transform_timm(is_train, args)
        print(f'TIMM transform for train ({is_train}):\n', transform)
    else:
        transform = build_transform(is_train, args)


    # if args.dataset_name == 'CIFAR':
    #     dataset = datasets.CIFAR100(args.dataset_root_path, train=is_train, transform=transform)
    #     dataset.num_classes = 100
    if args.dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root=args.dataset_root_path,
                              train=is_train,
                              transform=transform, download=True)
        dataset.dataset_name = 'cifar10'
        dataset.num_classes = 10
    elif args.dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root=args.dataset_root_path,
                               train=is_train,
                               transform=transform, download=True)
        dataset.dataset_name = 'cifar100'
        dataset.num_classes = 100
    elif args.dataset_image_folder:
        root = os.path.join(args.dataset_root_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        dataset.num_classes = len(np.unique(dataset.targets))
    elif args.dataset_in_memory and is_train:
        dataset = DatasetImgTargetInMemorySharded(args, split='train', transform=transform)
    elif args.dataset_in_memory:
        dataset = DatasetImgTargetInMemory(args, split='train' if is_train else 'val', transform=transform)
    else:
        dataset = DatasetImgTarget(args, split='train' if is_train else 'val', transform=transform)
    nb_classes = dataset.num_classes

    return dataset, nb_classes


def build_transform_timm(is_train, args):
    resize_im = args.input_size > 32
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.re_prob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )

        if args.pad_random_crop:
            transform.transforms[0] = RandomResizePadCrop(
                args.input_size, args.resize_size, padding_value=0)

        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)

        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.pad_random_crop:
            t.append(ResizeAndPad(args.test_input_size, padding_value=123))
        elif args.test_input_size >= 384:
            t.append(
                transforms.Resize((args.test_input_size, args.test_input_size), 
                                interpolation=transforms.InterpolationMode.BICUBIC), 
            )
            print(f"Warping {args.test_input_size} size input images...")
        else:
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(args.test_resize_size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.test_input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(t)
    return transform


def build_transform(is_train, args):
    input_size = args.input_size if is_train else args.test_input_size
    resize_size = args.resize_size if is_train else args.test_resize_size


    mean = MEANS['imagenet']
    std = STDS['imagenet']
    if args.custom_mean_std:
        mean = MEANS[args.dataset_name] if args.dataset_name in MEANS.keys() else MEANS['05']
        std = STDS[args.dataset_name] if args.dataset_name in STDS.keys() else STDS['05']


    t = []


    if args.transform_gpu and is_train:
        if args.pad_random_crop:
            t.append(RandomResizePadCrop(resize_size, resize_size, padding_value=0))
        if args.short_side_resize_random_crop:
            t.append(transforms.Resize(resize_size, interpolation=BICUBIC))
        else:
            # args.square_resize_random_crop
            t.append(transforms.Resize((resize_size, resize_size), interpolation=BICUBIC))

    elif is_train:
        if args.pad_random_crop:
            t.append(RandomResizePadCrop(input_size, resize_size, padding_value=0))
        elif args.short_side_resize_random_crop:
            t.append(transforms.Resize(
                resize_size, interpolation=BICUBIC))
            t.append(transforms.RandomCrop((input_size, input_size)))
        elif args.random_resized_crop:
            t.append(transforms.RandomResizedCrop((input_size, input_size), interpolation=BICUBIC))
        else:
            # args.square_resize_random_crop
            t.append(transforms.Resize(
                (resize_size, resize_size),
                interpolation=BICUBIC))
            t.append(transforms.RandomCrop(input_size))

        if args.horizontal_flip:
            t.append(transforms.RandomHorizontalFlip())

        if args.rand_aug:
            t.append(transforms.RandAugment())
        if args.trivial_aug:
            t.append(transforms.TrivialAugmentWide())

    else:
        if args.pad_random_crop:
            t.append(ResizeAndPad(input_size, padding_value=0))
        else:
            if args.short_side_resize_random_crop:
                t.append(transforms.Resize(resize_size, interpolation=BICUBIC))
            else:
                t.append(transforms.Resize((resize_size, resize_size), interpolation=BICUBIC))
            t.append(transforms.CenterCrop(input_size))


    if not args.transform_gpu:
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean=mean, std=std))


    if is_train and args.re_prob > 0 and not args.transform_gpu:
        # multiple random erasing blobs
        for _ in range(args.re_mult):
            t.append(
                transforms.RandomErasing(
                    p=args.re_prob, scale=(args.re_size_min, args.re_size_max), ratio=(args.re_r1, 3.3)
                )
            )


    transform = transforms.Compose(t)
    print(f'Transform for train ({is_train}):\n', transform)
    return transform

