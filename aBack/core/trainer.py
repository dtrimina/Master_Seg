#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
import numpy as np

import mmcv
from loguru import logger

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils import (
    setup_logger, get_model_info, save_checkpoint, load_ckpt, DictAverageMeter, SegMeter
)


# torch.autograd.set_detect_anomaly(True)


class Exp:

    def __init__(self):
        pass

    @staticmethod
    def get_model(name, cfg):
        if name == 'model_pld_and_seg':
            # old model pld_and_seg
            from models.models_pld_and_seg import Model_OD_and_Seg
            model = Model_OD_and_Seg(n_OD_classes=2, n_Seg_classes=16)
        elif name == 'model_baseline':
            from models.model_SEG import Model_Seg
            model = Model_Seg(n_classes=16)
        elif name == 'model_baseline_v1':
            from models.model_SEG_v1 import Model_Seg
            model = Model_Seg(n_classes=16)
        elif name == 'model_baseline_v2':
            from models.model_SEG_v2 import Model_Seg
            model = Model_Seg(n_classes=16)
        elif name == 'model_baseline_v3':
            from models.model_SEG_v3 import Model_Seg
            model = Model_Seg(n_classes=16)
        elif name == 'model_baseline_v3fullconv':
            from models.model_SEG_v3_fullconv import Model_Seg
            model = Model_Seg(n_classes=16)
        elif name == 'Space_RepVGGv3':
            from models.model_repvggv3 import Space_RepVGG
            model = Space_RepVGG()
        elif name == 'Space_RepVGGv3_bilinear':
            from models import Space_RepVGG
            model = Space_RepVGG()
        elif name == 'Space_RepVGGv4':
            from models.model_repvggv4 import Space_RepVGG
            model = Space_RepVGG()
        else:
            raise ValueError(f'model {name} not support.')

        return model

    @staticmethod
    def get_train_loader(cfg):
        from data import SegDataSet
        train_dataset = SegDataSet(cfg, mode='train')
        train_loader = data.DataLoader(train_dataset, batch_size=cfg.imgs_per_gpu, shuffle=True,
                                       num_workers=cfg.num_workers,
                                       pin_memory=True, drop_last=True)
        return train_loader

    @staticmethod
    def get_val_loader(cfg):
        from data import SegDataSet
        val_dataset = SegDataSet(cfg, mode='test')
        val_loader = data.DataLoader(val_dataset, batch_size=cfg.imgs_per_gpu, shuffle=False,
                                     num_workers=cfg.num_workers,
                                     pin_memory=True, drop_last=False)
        return val_loader

    @staticmethod
    def get_optimizer(model, cfg):  # todo code optimize as yolox

        # if cfg.model_name.startswith('STDCNet'):
        #     wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        #     param_list = [
        #         {'params': wd_params},
        #         {'params': nowd_params, 'weight_decay': 0},
        #         {'params': lr_mul_wd_params, 'lr_mul': True},
        #         {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr_mul': True},
        #     ]
        #     optimizer = optim.Adam(param_list, lr=cfg.lr, weight_decay=cfg.weight_decay)
        #
        #     return optimizer

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        optimizer = optim.Adam(pg0, lr=cfg.lr, weight_decay=cfg.weight_decay)
        optimizer.add_param_group(
            {"params": pg1, "weight_decay": cfg.weight_decay}
        )  # add pg1 with weight_decay
        optimizer.add_param_group({"params": pg2})
        return optimizer

    @staticmethod
    def get_lr_scheduler(cfg, optimizer):  # todo code optimize as yolox
        assert cfg.type_lr_scheduler in ['step_lr_scheduler', 'poly_lr_scheduler']

        if cfg.type_lr_scheduler == 'step_lr_scheduler':
            def burnin_schedule(i):
                if i < cfg.warmup_steps:
                    warmup_factor = (cfg.lr / cfg.warmup_start_lr) ** (1. / cfg.warmup_steps)
                    lr = cfg.warmup_start_lr * (warmup_factor ** i)
                    return lr
                stage = np.sum(np.asarray(cfg.steps) <= i)
                return 0.314 ** stage

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: burnin_schedule(x))
        elif cfg.type_lr_scheduler == 'poly_lr_scheduler':
            # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    lr_lambda=lambda step: (1 - step / cfg.max_step) ** cfg.lr_power)
        else:
            raise (f'{cfg.type_lr_scheduler} not support.')
        return scheduler

    @staticmethod
    def get_seg_loss_fn(model_name, cfg):
        if model_name == 'model_pld_and_seg':
            from toolbox.losses.model_loss import PLD_and_Seg_Loss
            loss_fn = PLD_and_Seg_Loss(n_classes=cfg.n_classes)
        elif model_name == 'model_baseline':
            from toolbox.losses.model_loss import BaseLineLoss
            loss_fn = BaseLineLoss(cfg=None)
        elif model_name == 'model_baseline_v1':
            from toolbox.losses.model_loss import BaseLineLossV1
            loss_fn = BaseLineLossV1(cfg=cfg)
        elif model_name == 'model_baseline_v2':
            from toolbox.losses.model_loss import BaseLineLossV2
            loss_fn = BaseLineLossV2(cfg=cfg)
        elif model_name == 'model_baseline_v3' or model_name == 'model_baseline_v3fullconv':
            from toolbox.losses.model_loss import BaseLineLossV3
            loss_fn = BaseLineLossV3(cfg=cfg)
        elif model_name == 'Space_RepVGGv3':
            from toolbox.losses.model_loss import RepVGGLossV3
            loss_fn = RepVGGLossV3(cfg)
        elif model_name == 'Space_RepVGGv3_bilinear':
            from toolbox.losses.model_loss import RepVGGLossV3
            loss_fn = RepVGGLossV3(cfg)
        elif model_name == 'Space_RepVGGv4':
            from toolbox.losses.model_loss import RepVGGLossV4
            loss_fn = RepVGGLossV4(cfg)
        elif model_name == 'model_pld_and_segv2':
            from toolbox.losses.model_loss import PLD_and_Seg_LossV2
            loss_fn = PLD_and_Seg_LossV2()
        elif model_name == 'model_pld_and_segv3':
            from toolbox.losses.model_loss import PLD_and_Seg_LossV3
            loss_fn = PLD_and_Seg_LossV3()
        else:
            raise ValueError(f'model_name {model_name}"s model_loss not defined.')
        return loss_fn


class Trainer:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args

        # training related attr
        # self.amp_training = args.fp16
        self.device = torch.device('cuda')

        # metric record
        self.meter = DictAverageMeter()

        self.logdir = f'run/Seg_{cfg.model_name}_{time.strftime("%Y-%m-%d-%H-%M")}'
        mmcv.mkdir_or_exist(self.logdir)

        mmcv.Config.dump(self.cfg, os.path.join(self.logdir, 'cfg.py'))  # fixme
        setup_logger(self.logdir, filename="train_log.log", mode="a", )
        logger.info(f"use logdir {self.logdir}")

        self.current_iter = 0
        self.start_iter = 0
        self.max_iter = self.cfg.max_step
        self.model_name = self.cfg.model_name
        self.print_step = self.cfg.print_step

        self.val_segmeter = SegMeter(n_classes=self.cfg.n_classes)

        # self.scaler = GradScaler()

    def train(self):
        """   load data_loader, loss_function, model ... before train """

        logger.info("args: {}".format(self.args))
        logger.info("cfg value:\n{}".format(self.cfg))

        # -----------------<<<<<<<<<<<<<<< model create  <<<<<<<<<<<<<<<--------------------
        # model related init
        self.model = Exp.get_model(name=self.model_name, cfg=self.cfg)

        self.model.to(self.device)
        self.resume_train(only_resume_model=True)
        self.model.train()
        logger.info("Model Summary: {}".format(
            get_model_info(self.model, (self.cfg.image_h, self.cfg.image_w))))  # fixme train size

        # solver related init
        self.optimizer = Exp.get_optimizer(model=self.model, cfg=self.cfg)
        self.lr_scheduler = Exp.get_lr_scheduler(self.cfg, self.optimizer)

        # -----------------<<<<<<<<<<<<<<< data loader  <<<<<<<<<<<<<<<--------------------
        self.train_loader = Exp.get_train_loader(cfg=self.cfg)
        self.train_iter = iter(self.train_loader)
        self.num_iter_per_epoch = len(self.train_loader)
        self.val_loader = Exp.get_val_loader(cfg=self.cfg)

        # -----------------<<<<<<<<<<<<<<< loss function <<<<<<<<<<<<<<--------------------
        self.loss_fn = Exp.get_seg_loss_fn(self.model_name, self.cfg)

        # -----------------<<<<<<<<<<<<<<< other <<<<<<<<<<<<<<<<<<<<<<<------------------
        self.tblogger = SummaryWriter(self.logdir)

        """  training """
        logger.info("Training start...")
        self.train_in_iter()

        logger.info("Training of experiment is done.")

    def train_in_iter(self):

        start_time = time.time()

        for index in range(self.max_iter):
            self.current_iter = index + 1
            # ----------------<<<<<<<<<<<<<<<<<<<< train one iter <<<<<<<<<<<<<<<<<------------------
            try:
                sample = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_loader)
                sample = next(self.train_iter)

            img = sample['img'].to(self.device)

            sample['label_cls'] = sample['label_cls'].to(self.device)
            sample['label_cls'].requires_grad = False

            # sample['label_freespace'] = sample['label_freespace'].to(self.device)
            # sample['label_freespace'].requires_grad = False
            sample['label_freespace'] = None

            sample['label_cls_encode'] = sample['label_cls_encode'].to(self.device)
            sample['label_cls_encode'].requires_grad = False

            self.optimizer.zero_grad()

            output_dict = self.model(img)
            loss_dict = self.loss_fn(output_dict, sample)

            self.meter.update(loss_dict)
            loss = loss_dict['loss']

            loss.backward()
            # clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)

            self.optimizer.step()
            self.lr_scheduler.step(self.current_iter)

            # ################## visialize during traing ##########################
            # imgs = img.cpu().numpy().transpose(0, 2, 3, 1)
            # segs = label_cls.cpu().numpy()
            # for index in range(imgs.shape[0]):
            #     img = (imgs[index] * 255.).astype(np.uint8)
            #     seg = segs[index]
            #
            #     img_seg = draw_seg_over_img(img, seg, PALETTE, n_classes=16, ignore_index=255,source_image_ration=0.4)
            #     cv2.imshow('result', img_seg)
            #     cv2.waitKey()
            # ############################################

            # -----------------------<<<<<<<<<<<<<<<<<< after iter <<<<<<<<<<<<<<<--------------------
            """
            `after_iter` contains two parts of logic:
                * log information
                * reset setting of resize
            """
            epoch = self.current_iter // self.num_iter_per_epoch

            if self.current_iter % self.print_step == 0:
                end_time = time.time()
                used_time = end_time - start_time
                eta_time = int((self.max_iter - self.current_iter) * (used_time / max(self.current_iter, 1)))
                used_time = str(datetime.timedelta(seconds=int(used_time)))
                eta_time = str(datetime.timedelta(seconds=eta_time))

                log_info = f' Iter |  step_{self.current_iter:7d} ep:{epoch:2d} | used_time: {used_time}, eta: {eta_time} | lr={self.lr_scheduler.get_lr()[0]:.7f} {self.meter.info()}'
                logger.info(log_info)

                self.save_ckpt()
                self.tblogger.add_scalar('lr', self.lr_scheduler.get_lr()[0], self.current_iter)
                for name in self.meter.sum_loss_dict:
                    self.tblogger.add_scalar(name, self.meter.get_avg(name), self.current_iter)

                self.meter.reset()

            if self.current_iter % self.num_iter_per_epoch == 0:
                logger.info(self.loss_fn.loss_summary.info)

            if self.current_iter % (self.num_iter_per_epoch * 5) == 0:
                # have trained 5 epoch
                self.validate(epoch)

            # if self.current_iter % self.print_step == 0:
            #     self.validate(epoch)

    def validate(self, epoch):
        logger.info('>>>>>>------------- val --------------->>>>>>')
        self.model.eval()
        self.val_segmeter.reset()
        with torch.no_grad():
            for sample in self.val_loader:
                img = sample['img'].to(self.device)
                label_cls = sample['label_cls'].cpu().numpy()

                pred_dict = self.model(img)

                pred_cls = pred_dict['pred_cls'].max(1)[1].cpu().numpy()

                self.val_segmeter.update(label_cls, pred_cls, ignore_fs_boundary=True)

            # out = {
            #     'PA': pixel_acc,
            #     'mPA': mPA,
            #     'mIoU': mIoU,
            #     'mBA': mBA,
            #
            #     'pa_per_cls': pixel_acc_per_cls,
            #     'iou_per_cls': iou_per_cls,
            #     'mBA_radius': self.mBA_boundary_radius,
            #     'ba_per_radiu': pixel_acc_per_radiu,
            #
            #     'pixel_ratio_per_class': pixel_ratio_per_class
            # }
            result = self.val_segmeter.get_scores()
            logger.info(f' Val | ep:{epoch}, PA: {result["PA"]}, mPA: {result["mPA"]}, mIoU: {result["mIoU"]}, mBA: {result["mBA"]}')

            for key in ['PA', 'mPA', 'mIoU', 'mBA']:
                self.tblogger.add_scalar(key, result[key], self.current_iter)

        self.model.train()

    def resume_train(self, only_resume_model=True):
        ckpt_file = self.cfg.pretrained
        if ckpt_file:
            logger.info(f"resume training from {ckpt_file}")

            ckpt = torch.load(ckpt_file, map_location=self.device)
            load_ckpt(self.model, ckpt['model'])

            if not only_resume_model:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self.start_iter = ckpt["start_iter"]
                logger.info(
                    "loaded checkpoint '{}' (iter {})".format(
                        self.args.resume, self.current_iter
                    )
                )

    def save_ckpt(self):
        save_model = self.model
        # logger.info("Save weights to {}".format(self.logdir))
        ckpt_state = {
            "start_iter": self.current_iter + 1,
            "model": save_model.state_dict(),  # fixme
            "optimizer": self.optimizer.state_dict(),
        }
        # if self.amp_training:
        #     # save amp state according to
        #     # https://nvidia.github.io/apex/amp.html#checkpointing
        #     ckpt_state["amp"] = amp.state_dict()
        save_checkpoint(
            ckpt_state,
            self.logdir,
        )
