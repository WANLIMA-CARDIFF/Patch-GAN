import os
import random
import shutil
import yaml
from attrdict import AttrMap
import time

import torch
from torch import nn
from torch.backends import cudnn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision.transforms as transforms
from data_manager import TrainDataset
from models.gen.SPANet import Generator, MyGenerator
from models.dis.dis import Discriminator, PatchDiscriminator
import utils
from utils import gpu_manage, save_image, checkpoint
from eval import test
from log_report import LogReport
from log_report import TestReport
import util.lr_decay as lrd

import models_mae


def train(config):
    gpu_manage(config)

    ### DATASET LOAD ###
    print('===> Loading datasets')
    transform_list = [
        transforms.ToTensor(),
    ]
    transform_train = transforms.Compose(transform_list)
    dataset = TrainDataset(config, transform = transform_train)
    print('dataset:', len(dataset))
    train_size = int((1 - config.validation_size) * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
    print('train dataset:', len(train_dataset))
    print('validation dataset:', len(validation_dataset))
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=config.threads, batch_size=config.batchsize, shuffle=True)
    validation_data_loader = DataLoader(dataset=validation_dataset, num_workers=config.threads, batch_size=config.validation_batchsize, shuffle=False)
    
    ### MODELS LOAD ###
    print('===> Loading models')
    if config.Generator.startswith("mae_vit"):
        gen = models_mae.__dict__[config.Generator](norm_pix_loss=False)
        # load pre-trained model
        checkpoint_model = torch.load(config.pretrained_model)
        msg = gen.load_state_dict(checkpoint_model['model'], strict=False)
        print(msg)
    else:
        raise ValueError('wrong model')

    if config.gen_init is not None:
        param = torch.load(config.gen_init)
        gen.load_state_dict(param)
        print('load {} as pretrained model'.format(config.gen_init))

    patch_discriminator = PatchDiscriminator()

    param_groups = lrd.param_groups_lrd(gen, 0.05,
        no_weight_decay_list=gen.no_weight_decay(),
        layer_decay=0.75
    )
    opt_gen = torch.optim.AdamW(param_groups, lr=config.lr*config.batchsize/512)
 
    optimizer_p_dis = optim.Adam(patch_discriminator.parameters(), lr=config.lr_dis, betas=(0.5, 0.999))

    
    M = torch.FloatTensor(config.batchsize, config.width, config.height)

    criterionL1 = nn.L1Loss()
    criterionMSE = nn.MSELoss()
    criterionSoftplus = nn.Softplus()

    if config.cuda:
        gen = gen.cuda()
        patch_discriminator = patch_discriminator.cuda()
        criterionL1 = criterionL1.cuda()
        criterionMSE = criterionMSE.cuda()
        criterionSoftplus = criterionSoftplus.cuda()
        
        M = M.cuda()

    # 损失函数
    criterion_gan = nn.BCELoss()  # GAN 损失

    # 标签
    real_label = 1.0
    fake_label = 0.0
    max_ssim = 0
    logreport = LogReport(log_dir=config.out_dir)
    validationreport = TestReport(log_dir=config.out_dir)

    print('===> begin')
    start_time=time.time()
    # main
    for epoch in range(1, config.epoch + 1):
        epoch_start_time = time.time()
        for iteration, batch in enumerate(training_data_loader, 1):
            if config.cuda:
                real_a, real_b  = batch[0].to("cuda"), batch[1].to("cuda")#, batch[2].to("cuda")
            else:
                real_a, real_b  = batch[0], batch[1]

            random_config_mask_rate = random.uniform(0,  config.mask_rate)
            fake_b, loss_mse, mask, pred_patched, GT_patched  = gen.forward(real_a, real_b, random_config_mask_rate)

            optimizer_p_dis.zero_grad()

            # 判别真实图像
            real_preds = patch_discriminator(real_b)  # [B, num_patches]
            real_targets = torch.ones_like(real_preds) * real_label
            loss_D_real = criterion_gan(real_preds, real_targets)

            # 判别生成的图像
            # reconstructed = generator(batch['input'])  # 用生成器生成图像
            fake_preds = patch_discriminator(fake_b.detach())  # detach 避免更新生成器
            fake_targets = torch.ones_like(fake_preds) * fake_label
            loss_D_fake = criterion_gan(fake_preds, fake_targets)

            # 总鉴别器损失
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            optimizer_p_dis.step()

            ################
            ### Update G ###
            ################

            opt_gen.zero_grad()

            loss_g_l1 = criterionL1(fake_b, real_b) * config.lamb

            ################
            ### Update G2 ##
            ################
            fake_preds = patch_discriminator(fake_b)
            gan_targets = torch.ones_like(fake_preds) * real_label  # 生成器希望判别器认为是“真实”
            loss_G_gan = criterion_gan(fake_preds, gan_targets)
            loss_g = loss_g_l1 + (loss_mse*100) + loss_G_gan #+ disc_loss+ adv_loss # + loss_g_att
            loss_g.backward()

            opt_gen.step()


            # log
            if iteration % 10 == 0:
                print("===> Epoch[{}]({}/{}):    loss_g_l1: {:.4f} loss_mse: {:.4f} loss_D: {:.4f} loss_G_gan: {:.4f}".format(
                epoch, iteration, len(training_data_loader),  loss_g_l1.item(), loss_mse.item(), loss_D.item(), loss_G_gan.item()))
                
                log = {}
                log['epoch'] = epoch
                log['iteration'] = len(training_data_loader) * (epoch-1) + iteration
                log['gen/loss'] = loss_g.item()

                logreport(log)

        print('epoch', epoch, 'finished, use time', time.time() - epoch_start_time)
        with torch.no_grad():
            log_validation = test(config, validation_data_loader, gen, criterionMSE, epoch)
            validationreport(log_validation)
        print('validation finished')
        if log_validation['ssim'] > max_ssim and epoch > 5:
            checkpoint(config, "best", gen, patch_discriminator)
            max_ssim = log_validation['ssim']
        if epoch == 200 :
            checkpoint(config, "last", gen, patch_discriminator)
        validationreport.save_lossgraph()

    print('training time:', time.time() - start_time)


if __name__ == '__main__':
    with open('config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    utils.make_manager()
    n_job = utils.job_increment()
    config.out_dir = os.path.join(config.out_dir, '{:06}'.format(n_job))
    os.makedirs(config.out_dir)
    print('Job number: {:04d}'.format(n_job))

    # 保存本次训练时的配置
    shutil.copyfile('config.yml', os.path.join(config.out_dir, 'config.yml'))

    train(config)
