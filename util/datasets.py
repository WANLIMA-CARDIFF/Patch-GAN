# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import numpy as np
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import os
from typing import Any, Callable, Optional, Tuple, List, Dict
from PIL import Image
from torchvision.datasets.vision import VisionDataset

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class ImageFolder(VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.loader = loader if loader is not None else self.default_loader
        self.extensions = None if is_valid_file else IMG_EXTENSIONS
        self.samples = self.make_dataset(self.root, self.extensions, is_valid_file)
        self.classes, self.class_to_idx = self._find_classes(self.root)
        self.targets = [s[1] for s in self.samples]

    @staticmethod
    def make_dataset(
        directory: str,
        extensions: Optional[List[str]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None
    ) -> List[Tuple[str, int]]:
        instances = []
        directory = os.path.expanduser(directory)

        for target_class in sorted(os.listdir(directory)):
            class_index = -1
            target_dir = os.path.join(directory, target_class)

            if not os.path.isdir(target_dir):
                continue

            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    path_GT = os.path.join(root, fname).replace("input/cloud","output/label")
                    if is_valid_file is not None:
                        if is_valid_file(path):
                            instances.append((path, path_GT, class_index))
                    elif extensions is not None and path.lower().endswith(tuple(extensions)):
                        instances.append((path, path_GT, class_index))

        return instances

    @staticmethod
    def _find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, path_GT, target = self.samples[index]
        sample = self.loader(path)
        label = self.loader(path_GT)
        if self.transform is not None:
            sample = self.transform(sample)
            label = self.target_transform(label)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        # print("777777777777777777")
        # print(np.asarray(sample).max())
        # print(np.asarray(sample).min())
        return sample, label, target

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def default_loader(path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

# List of supported image extensions
IMG_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'
]

# Example Usage:
# dataset_train = ImageFolder('path_to_data/train', transform=transform_train)