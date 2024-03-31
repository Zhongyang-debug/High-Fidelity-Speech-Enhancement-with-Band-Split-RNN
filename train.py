#!/usr/bin/env python
# coding: utf-8

import os
import dataloader
import torch
import torch.nn.functional as F
import logging
from torchinfo import summary
import argparse
from natsort import natsorted
import librosa
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from module import *
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


class Trainer:

    def __init__(self, train_ds, test_ds, args, rank, device):

        self.n_fft = 512
        self.hop = 128
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.args = args
        self.rank = rank
        self.device = device
        
        self.model = BSRNN(num_channel=64, num_layer=5).to(self.device)
#        summary(self.model, [(1, 257, args.cut_len//self.hop+1, 2)])
        self.discriminator = Discriminator(ndf=16).to(self.device)
#        summary(self.discriminator, [(1, 1, int(self.n_fft/2)+1, args.cut_len//self.hop+1),
#                                     (1, 1, int(self.n_fft/2)+1, args.cut_len//self.hop+1)])

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.init_lr)
        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.init_lr)

        if torch.cuda.device_count() > 1:
            self.model = DDP(self.model, device_ids=[rank]).to(self.device)
            self.discriminator = DDP(self.discriminator, device_ids=[rank]).to(self.device)

    def train_step(self, batch, use_disc):

        clean = batch[0].to(self.device)
        noisy = batch[1].to(self.device)
        one_labels = torch.ones(clean.size(0)).to(self.device)
    
        self.optimizer.zero_grad()

        noisy_spec = torch.stft(noisy,
                                n_fft=self.n_fft,
                                hop_length=self.hop,
                                window=torch.hann_window(self.n_fft).to(self.device),
                                onesided=True,
                                return_complex=True)
        clean_spec = torch.stft(clean,
                                n_fft=self.n_fft,
                                hop_length=self.hop,
                                window=torch.hann_window(self.n_fft).to(self.device),
                                onesided=True,
                                return_complex=True)
                
        est_spec = self.model(noisy_spec)

        est_mag = (torch.abs(est_spec).unsqueeze(1) + 1e-10) ** 0.3
        clean_mag = (torch.abs(clean_spec).unsqueeze(1) + 1e-10) ** 0.3
        noisy_mag = (torch.abs(noisy_spec).unsqueeze(1) + 1e-10) ** 0.3
        
        mae_loss = nn.L1Loss()
        loss_mag = mae_loss(est_mag, clean_mag)
        loss_ri = mae_loss(est_spec, clean_spec)

        if use_disc is False:
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag
        else:
            predict_fake_metric = self.discriminator(clean_mag, est_mag)
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * gen_loss_GAN

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
        self.optimizer.step()
        
        est_audio = torch.istft(est_spec,
                                n_fft=self.n_fft,
                                hop_length=self.hop,
                                window=torch.hann_window(self.n_fft).to(self.device),
                                onesided=True)

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy())
        noisy_audio_list = list(noisy.cpu().numpy())
        pesq_score = batch_pesq(clean_audio_list, est_audio_list)
        pesq_score_n = batch_pesq(est_audio_list, noisy_audio_list)

        # The calculation of PESQ can be None due to silent part
        if pesq_score is not None and pesq_score_n is not None:
            self.optimizer_disc.zero_grad()
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            predict_min_metric = self.discriminator(est_mag.detach(), noisy_mag)            
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels.float()) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score) + \
                                  F.mse_loss(predict_min_metric.flatten(), pesq_score_n)
            discrim_loss_metric.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5)
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.])
                
        return loss.item(), discrim_loss_metric.item()

    @torch.no_grad()
    def test_step(self, batch, use_disc):

        clean = batch[0].to(self.device)
        noisy = batch[1].to(self.device)
        one_labels = torch.ones(clean.size(0)).to(self.device)

        noisy_spec = torch.stft(noisy,
                                n_fft=self.n_fft,
                                hop_length=self.hop,
                                window=torch.hann_window(self.n_fft).to(self.device),
                                onesided=True,
                                return_complex=True)
        clean_spec = torch.stft(clean,
                                n_fft=self.n_fft,
                                hop_length=self.hop,
                                window=torch.hann_window(self.n_fft).to(self.device),
                                onesided=True,
                                return_complex=True)
        
        est_spec = self.model(noisy_spec)

        est_mag = (torch.abs(est_spec).unsqueeze(1) + 1e-10) ** 0.3
        clean_mag = (torch.abs(clean_spec).unsqueeze(1) + 1e-10) ** 0.3
        noisy_mag = (torch.abs(noisy_spec).unsqueeze(1) + 1e-10) ** 0.3

        mae_loss = nn.L1Loss()
        loss_mag = mae_loss(est_mag, clean_mag)
        loss_ri = mae_loss(est_spec, clean_spec)

        if use_disc is False:
            loss = self.args.loss_weights[0] * loss_ri + self.args.loss_weights[1] * loss_mag
        else:
            predict_fake_metric = self.discriminator(clean_mag, est_mag)
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())
            loss = (self.args.loss_weights[0] * loss_ri +
                    self.args.loss_weights[1] * loss_mag +
                    self.args.loss_weights[2] * gen_loss_GAN)

        est_audio = torch.istft(est_spec,
                                n_fft=self.n_fft,
                                hop_length=self.hop,
                                window=torch.hann_window(self.n_fft).to(self.device),
                                onesided=True)

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy())
        noisy_audio_list = list(noisy.cpu().numpy())
        pesq_score = batch_pesq(clean_audio_list, est_audio_list)
        pesq_score_n = batch_pesq(est_audio_list, noisy_audio_list)
        if pesq_score is not None and pesq_score_n is not None:
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            predict_min_metric = self.discriminator(est_mag.detach(), noisy_mag)            
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score) + \
                                  F.mse_loss(predict_min_metric.flatten(), pesq_score_n)
        else:
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item()

    def test(self, use_disc):

        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.
        disc_loss_total = 0.
        for idx, batch in enumerate(tqdm(self.test_ds)):
            step = idx + 1
            loss, disc_loss = self.test_step(batch, use_disc)
            gen_loss_total += loss
            disc_loss_total += disc_loss
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step

        template = 'Generator loss: {}, Discriminator loss: {}'
        logging.info(template.format(gen_loss_avg, disc_loss_avg))

        return gen_loss_avg

    def train(self):

        scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.decay_epoch, gamma=0.98)
        scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_disc, step_size=self.args.decay_epoch, gamma=0.98)
        for epoch in range(self.args.epochs):
            self.model.train()
            self.discriminator.train()

            loss_total = 0
            loss_gan = 0
            
            if epoch >= (self.args.epochs/2):
                use_disc = True
            else:
                use_disc = False
            
            for idx, batch in enumerate(tqdm(self.train_ds)):
                step = idx + 1
                loss, disc_loss = self.train_step(batch, use_disc)
                template = 'Epoch {}, Step {}, loss: {}, disc_loss: {}'
                
                loss_total = loss_total + loss
                loss_gan = loss_gan + disc_loss
                
                if (step % args.log_interval) == 0:
                    logging.info(template.format(epoch, step, loss_total/step, loss_gan/step))

            gen_loss = self.test(use_disc)
            path = os.path.join(self.args.save_model_dir, 'gene_epoch_' + str(epoch) + '_' + str(gen_loss)[:5])
            path_d = os.path.join(self.args.save_model_dir, 'disc_epoch_' + str(epoch))
            os.makedirs(self.args.save_model_dir, exist_ok=True)
            if 0 == self.rank:
                torch.save(self.model.state_dict(), path)
                torch.save(self.discriminator.state_dict(), path_d)
            scheduler_G.step()
            scheduler_D.step()


def main(rank: int, world_size: int, args: list):

    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"  # 主节点的地址
        os.environ["MASTER_PORT"] = "12355"  # 主节点用于通信的端口号
        init_process_group(
            backend="nccl",  # nccl 是一个针对 NVIDIA GPU 的通信库, 专为深度学习等计算密集型任务设计
            rank=rank,  # 标识当前进程的位置
            world_size=world_size  # 总进程数，确保所有参与分布式训练的进程都已被考虑
        )  # 初始化分布式环境

    device = torch.device('cuda:{:d}'.format(rank) if torch.cuda.is_available() else 'cpu')
    train_ds, test_ds = dataloader.load_data(args.data_dir, args.batch_size, 4, args.cut_len, rank)
    trainer = Trainer(train_ds, test_ds, args, rank, device)
    trainer.train()

    if world_size > 1:
        # 清理与进程组相关的所有资源
        destroy_process_group()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",
                        type=int,
                        default=120,
                        help="number of epochs of training")
    parser.add_argument("--batch_size",
                        type=int,
                        default=6)
    parser.add_argument("--log_interval",
                        type=int,
                        default=500)
    parser.add_argument("--decay_epoch",
                        type=int,
                        default=10,
                        help="epoch from which to start lr decay")
    parser.add_argument("--init_lr",
                        type=float,
                        default=1e-3,
                        help="initial learning rate")
    parser.add_argument("--cut_len",
                        type=int,
                        default=int(16000 * 2),
                        help="cut length, default is 2 seconds in denoise and dereverberation")
    parser.add_argument("--data_dir",
                        type=str,
                        default='D:/Audio_Project/Speech_Enhancement/Code/Speech_Enhancement/dataset',
                        help="dir of VCTK+DEMAND dataset")
    parser.add_argument("--save_model_dir",
                        type=str,
                        default='./saved_model',
                        help="dir of saved model")
    parser.add_argument("--loss_weights",
                        type=list,
                        default=[0.5, 0.5, 1],
                        help="weights of RI components, magnitude, and Metric Disc")
    args, _ = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO)

    # 随机数种子
    np.random.seed(123)
    torch.manual_seed(123)
    num_gpus = 0
    multiprocess = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        if multiprocess:
            num_gpus = torch.cuda.device_count()

    if num_gpus > 1 and multiprocess:
        # GPUs
        mp.spawn(fn=main,  # 函数作为派生进程的入口点被调用
                 args=(num_gpus, args),  # 传递给函数的参数
                 nprocs=num_gpus,  # 创建的进程数
                 join=True,  # 主进程将等待所有派生进程完成后再继续执行
                 daemon=False,  # 派生进程不是守护进程，即主进程结束时它们不会自动结束
                 start_method='spawn')  # 启动一个新的Python解释器来执行进程
    else:
        # GPU or CPU
        main(0, num_gpus, args)


