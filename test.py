import numpy as np
from skimage.metrics import structural_similarity as SSIM

import os
import random
import shutil
import yaml
from attrdict import AttrMap
import time
from torch.autograd import Variable

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision.transforms as transforms
from data_manager import TestDataset
from models.gen.SPANet import Generator
import utils
from utils import gpu_manage, save_image, checkpoint
import models_mae
from niqe.niqe import niqe

def sliding_window_reconstruction(model, image, GT, crop_size=224, stride=112, device='cuda'):
    C, H, W = image.shape  # 获取图像的通道数、高度和宽度
    image = image.to(device)  # 将图像移到设备上
    GT = GT.to(device)  # 初始化输出重建图像和权重计数
    reconstructed_image = torch.zeros_like(image, device=device)  # (C, H, W)
    weight_map = torch.zeros((1, H, W), device=device)  # 用于记录每个像素被处理的次数

    # 滑动窗口遍历
    for y in range(0, H - crop_size + 1, stride):
        for x in range(0, W - crop_size + 1, stride):
            # 裁剪图像块
            crop = image[:, y:y+crop_size, x:x+crop_size].unsqueeze(0)  # (1, C, crop_size, crop_size)
            
            # 模型推理
            with torch.no_grad():
                reconstructed_crop = model(crop, GT, split="test")  # (1, C, crop_size, crop_size)

            # 累加重建结果
            reconstructed_image[:, y:y+crop_size, x:x+crop_size] += reconstructed_crop.squeeze(0)
            weight_map[:, y:y+crop_size, x:x+crop_size] += 1

    # 处理边界 (确保图像尺寸完整覆盖)
    if H % stride != 0:
        for x in range(0, W - crop_size + 1, stride):
            crop = image[:, H-crop_size:H, x:x+crop_size].unsqueeze(0)
            with torch.no_grad():
                reconstructed_crop = model(crop, GT, split="test")
            reconstructed_image[:, H-crop_size:H, x:x+crop_size] += reconstructed_crop.squeeze(0)
            weight_map[:, H-crop_size:H, x:x+crop_size] += 1

    if W % stride != 0:
        for y in range(0, H - crop_size + 1, stride):
            crop = image[:, y:y+crop_size, W-crop_size:W].unsqueeze(0)
            with torch.no_grad():
                reconstructed_crop = model(crop, GT, split="test")
            reconstructed_image[:, y:y+crop_size, W-crop_size:W] += reconstructed_crop.squeeze(0)
            weight_map[:, y:y+crop_size, W-crop_size:W] += 1

    if H % stride != 0 and W % stride != 0:
        crop = image[:, H-crop_size:H, W-crop_size:W].unsqueeze(0)
        with torch.no_grad():
            reconstructed_crop = model(crop, GT, split="test")
        reconstructed_image[:, H-crop_size:H, W-crop_size:W] += reconstructed_crop.squeeze(0)
        weight_map[:, H-crop_size:H, W-crop_size:W] += 1

    # 对每个像素进行平均（加权融合）
    reconstructed_image /= weight_map

    return reconstructed_image

def test(config, test_data_loader, gen, epoch=1):
    criterionMSE = nn.MSELoss()
    avg_mse = 0
    avg_psnr = 0
    avg_ssim = 0
    # avg_niqe = 0
    for i, batch in enumerate(test_data_loader):
        x, t = torch.Tensor(Variable(batch[0])), torch.Tensor(Variable(batch[1]))
        if config.cuda:
            x = x.cuda(0)
            t = t.cuda(0)

        with torch.no_grad():
            x = x.squeeze(0)
            out = sliding_window_reconstruction(gen, x, t, crop_size=224, stride=112, device='cuda')
            out = out.unsqueeze(0)
        if epoch % config.snapshot_interval == 0:
            h = 1
            w = 3
            c = 3
            width = config.width
            height = config.height

            allim = np.zeros((h, w, c, width, height))
            x_ = x.cpu().numpy()[0]
            t_ = t.cpu().numpy()[0]
            out_ = out.cpu().numpy()[0]
            in_rgb = x_[:3]
            t_rgb = t_[:3]
            out_rgb = np.clip(out_[:3], 0, 1)
            allim[0, 0, :] = in_rgb * 255
            allim[0, 1, :] = out_rgb * 255
            allim[0, 2, :] = t_rgb * 255

            allim = allim.transpose(0, 3, 1, 4, 2)
            allim = allim.reshape((h*height, w*width, c))

            save_image(config.out_dir, allim, i, epoch)

        mse = criterionMSE(out, t)
        psnr = 10 * np.log10(1 / mse.item())
     
        img1 = np.tensordot(out.cpu().numpy()[0, :3].transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
        img2 = np.tensordot(t.cpu().numpy()[0, :3].transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)

        ssim = SSIM(img1, img2)

        avg_mse += mse.item()
        avg_psnr += psnr
        avg_ssim += ssim

    avg_mse = avg_mse / len(test_data_loader)
    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)


    # print("===> Avg. MSE: {:.4f}".format(avg_mse))
    # print("===> Avg. MAE: {:.4f}".format(np.sqrt(avg_mse)))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    print("===> Avg. SSIM: {:.4f} dB".format(avg_ssim))
    # print("===> Avg. niqe:", avg_niqe)

    log_test = {}
    log_test['epoch'] = epoch
    log_test['mse'] = avg_mse
    log_test['psnr'] = avg_psnr
    log_test['ssim'] = avg_ssim

    return log_test

if __name__ == '__main__':
    with open(os.path.join("./", 'config.yml'), 'r', encoding='UTF-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    gpu_manage(config)

    ### DATASET LOAD ###
    print('===> Loading datasets')
    transform_list = [
        transforms.ToTensor(),
    ]
    transform = transforms.Compose(transform_list)
    dataset = TestDataset(config, transform=transform)
    print('dataset:', len(dataset))
    test_data_loader = DataLoader(dataset=dataset, num_workers=config.threads, batch_size=config.batchsize, shuffle=False)

    if config.Generator.startswith("mae_vit"):
        gen = models_mae.__dict__[config.Generator](norm_pix_loss=False)
        # load pre-trained model
        checkpoint_model = torch.load(os.path.join("./models/rice1.pth"))
        msg = gen.load_state_dict(checkpoint_model, strict=False)
        if config.cuda:
            gen = gen.cuda()
        print(msg)

        test(config, test_data_loader, gen, )
