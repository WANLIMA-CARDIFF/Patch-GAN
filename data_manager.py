import glob
import cv2
import random
import numpy as np
import pickle
import os
from PIL import Image

from torch.utils import data


class TrainDataset(data.Dataset):

    def __init__(self, config, transform):
        super().__init__()
        self.config = config
        self.transform = transform
        self.randomcropcv = RandomCropCV((config.width, config.height))
        train_list_file = os.path.join(config.datasets_dir, config.train_list)
        # 如果数据集尚未分割，则进行训练集和测试集的分割
        if not os.path.exists(train_list_file) or os.path.getsize(train_list_file) == 0:
            files = os.listdir(os.path.join(config.datasets_dir, 'ground_truth'))
            random.shuffle(files)
            n_train = int(config.train_size * len(files))
            train_list = files[:n_train]
            test_list = files[n_train:]
            np.savetxt(os.path.join(config.datasets_dir, config.train_list), np.array(train_list), fmt='%s')
            np.savetxt(os.path.join(config.datasets_dir, config.test_list), np.array(test_list), fmt='%s')

        self.imlist = np.loadtxt(train_list_file, str)

    def __getitem__(self, index):
        
        t = cv2.imread(os.path.join(self.config.datasets_dir, 'ground_truth', str(self.imlist[index])), 1)#.astype(np.float32)
        x = cv2.imread(os.path.join(self.config.datasets_dir, 'cloudy_image', str(self.imlist[index])), 1)#.astype(np.float32)
        t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)  # Convert to RGB
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)  # Convert to RGB
        # t = cv2.resize(t, (self.config.width, self.config.height))
        # x = cv2.resize(x, (self.config.width, self.config.height))
        
        # M = np.clip((t-x).sum(axis=2), 0, 1).astype(np.float32)
        # x = x / 255.0
        # t = t / 255.0
        # x = x.transpose(2, 0, 1)
        # t = t.transpose(2, 0, 1)
        # x = Image.fromarray(x)
        # t = Image.fromarray(t)
        # M = Image.fromarray(M)

        if self.transform is not None:
            x,t = self.randomcropcv(x,t)
            x = self.transform(x)
            t = self.transform(t)
            # M = self.transform(M)
        return x, t#, M

    def __len__(self):
        return len(self.imlist)


class TestDataset(data.Dataset):

    def __init__(self, config, transform):
        super().__init__()
        self.config = config
        self.transform = transform
        self.randomcropcv = RandomCropCV((config.width, config.height))
        test_list_file = os.path.join(config.datasets_dir, config.test_list)
        # 如果数据集尚未分割，则进行训练集和测试集的分割
        if not os.path.exists(test_list_file) or os.path.getsize(test_list_file) == 0:
            files = os.listdir(os.path.join(config.datasets_dir, 'ground_truth'))
            random.shuffle(files)
            n_train = int(config.train_size * len(files))
            test_list = files[:n_train]
            test_list = files[n_train:]
            np.savetxt(os.path.join(config.datasets_dir, config.test_list), np.array(test_list), fmt='%s')
            np.savetxt(os.path.join(config.datasets_dir, config.test_list), np.array(test_list), fmt='%s')

        self.imlist = np.loadtxt(test_list_file, str)

    def __getitem__(self, index):

        t = cv2.imread(os.path.join(self.config.datasets_dir, 'ground_truth', str(self.imlist[index])), 1)  # .astype(np.float32)
        x = cv2.imread(os.path.join(self.config.datasets_dir, 'cloudy_image', str(self.imlist[index])), 1)  # .astype(np.float32)
        t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)  # Convert to RGB
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)  # Convert to RGB
        # t = cv2.resize(t, (self.config.width, self.config.height))
        # x = cv2.resize(x, (self.config.width, self.config.height))
        # x,t = self.randomcropcv(x,t)
        # M = np.clip((t - x).sum(axis=2), 0, 1).astype(np.float32)
        # x = x / 255.0
        # t = t / 255.0
        # x = x.transpose(2, 0, 1)
        # t = t.transpose(2, 0, 1)
        # x = Image.fromarray(x)
        # t = Image.fromarray(t)
        # M = Image.fromarray(M)

        if self.transform is not None:
            x = self.transform(x)
            t = self.transform(t)
            # M = self.transform(M)
        return x, t#, M

    def __len__(self):
        return len(self.imlist)



class RandomCropCV:
    def __init__(self, crop_size):
        """
        初始化随机裁剪操作。
        Args:
            crop_size (tuple): 裁剪的目标尺寸 (height, width)。
        """
        self.crop_size = crop_size

    def __call__(self, img, gt):
        """
        对 img 和 gt 同时进行相同的随机裁剪。
        Args:
            img (np.array): 输入图像，形状 (H, W, C)。
            gt (np.array): Ground truth，形状 (H, W) 或 (H, W, C)。
        Returns:
            tuple: 裁剪后的 (img, gt)。
        """
        h, w = img.shape[:2]
        crop_h, crop_w = self.crop_size

        if h < crop_h or w < crop_w:
            raise ValueError("Crop size must be smaller than image size.")

        # 随机生成裁剪起点
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        # 裁剪图像
        img_cropped = img[top:top + crop_h, left:left + crop_w,:]
        gt_cropped = gt[top:top + crop_h, left:left + crop_w,:]

        return img_cropped, gt_cropped