import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict

from models.layers import CBR
from models.models_utils import weights_init, print_network

from timm.models.vision_transformer import PatchEmbed, Block
import timm

class _Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch

        self.c0_0 = CBR(in_ch, 32, bn=False, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c0_1 = CBR(out_ch, 32, bn=False, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c1 = CBR(64, 128, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c2 = CBR(128, 256, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c3 = CBR(256, 512, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c4 = nn.Conv2d(512, 1, 3, 1, 1)

    def forward(self, x):
        x_0 = x[:, :self.in_ch]
        x_1 = x[:, self.in_ch:]
        h = torch.cat((self.c0_0(x_0), self.c0_1(x_1)), 1)
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        return h


class Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch, gpu_ids):
        super().__init__()
        self.gpu_ids = gpu_ids

        self.dis = nn.Sequential(OrderedDict([('dis', _Discriminator(in_ch, out_ch))]))

        self.dis.apply(weights_init)

    def forward(self, x):
        if self.gpu_ids:
            return nn.parallel.data_parallel(self.dis, x, self.gpu_ids)
        else:
            return self.dis(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super(PatchDiscriminator, self).__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        # 判别器头
        self.patch_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def forward(self, x):
        """
        x: 输入图像，形状为 [B, C, H, W]
        """
        # print(self.vit)
        # print(x.size())
        # features = self.vit(x)  # 形状 [B, num_patches, embed_dim]
        # print("features", features.size())
        patched_img = self.patchify(x)
        # print("patched_img:", patched_img.size())
        patch_outputs = self.patch_head(patched_img)  # 形状 [B, num_patches, 1]
        # print("patch_outputs", patch_outputs.size())
        return patch_outputs.squeeze(-1)  # 输出形状 [B, num_patches]

# class PatchDiscriminator(nn.Module):
#     def __init__(self, embed_dim=768):
#         super(PatchDiscriminator, self).__init__()
#         # 使用 timm 加载 ViT 模型
#         self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
#         self.vit.head = nn.Identity()  # 移除分类头
#
#         # 判别器头
#         self.patch_head = nn.Sequential(
#             nn.Linear(embed_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         """
#         x: 输入图像，形状为 [B, C, H, W]
#         """
#         features = self.vit(x)  # 形状 [B, num_patches, embed_dim]
#         print("features:", features.size())
#         patch_outputs = self.patch_head(features)  # 形状 [B, num_patches, 1]
#         return patch_outputs.squeeze(-1)  # 输出形状 [B, num_patches]