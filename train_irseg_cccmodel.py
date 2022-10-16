import os
import shutil
import time

import torch
import torch.nn as nn
import numpy as np

import mmcv
from loguru import logger

from utils.logger import setup_logger
from utils.metrics import averageMeter, runningScore
from utils.checkpoint import load_ckpt, save_ckpt

from common import N_CLASSES
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# from toolbox import MscCrossEntropyLoss
# from toolbox import get_logger
# from toolbox import averageMeter, runningScore
# from toolbox import ClassWeight, save_ckpt
# from toolbox import load_ckpt
# from toolbox import CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, ProbOhemCrossEntropy2d, FocalLoss2d, \
#     LovaszSoftmax, LDAMLoss


def run(cfg_path):
    device = torch.device('cuda')

    ####################
    #  加载配置文件信息
    ####################
    cfg = mmcv.Config.fromfile(cfg_path)

    ####################
    # 在run目录下新建logdir，用于保存当前训练模型、log等信息
    ####################
    logdir = f'run/{cfg.model_name}_{time.strftime("%Y-%m-%d-%H-%M")}'
    mmcv.mkdir_or_exist(logdir)
    shutil.copy(cfg_path, os.path.join(logdir, 'cfg.py'))

    ####################
    # 获取logger对象，用于后续记录训练过程中训练loss，测试指标等信息
    ####################
    setup_logger(logdir, filename="train_log.log", mode="a", )
    logger.info(f"use logdir {logdir}")

    ####################
    # 定义模型
    ####################
    from models.cccmodel import MobileDual
    model = MobileDual(n_classes=N_CLASSES)
    model.to(device)
    # 加载预训练模型
    # model.load_state_dict(torch.load('/home/space/Projects/SegProject/Segmentation_final/run/2022-09-27-15-52(irseg-cccmodel)/model.pth'))

    ####################
    # 导入训练集、测试集
    ####################
    from torch.utils.data import DataLoader
    from data.irseg import IRSeg
    train_dataset, test_dataset = IRSeg(cfg, mode='train'), IRSeg(cfg, mode='test')
    train_loader = DataLoader(train_dataset, batch_size=cfg.imgs_per_gpu, shuffle=True,
                                   num_workers=cfg.num_workers,
                                   pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.imgs_per_gpu, shuffle=False,
                                 num_workers=cfg.num_workers,
                                 pin_memory=True, drop_last=False)


    ####################
    # 定义优化器、学习率衰减策略
    ####################
    from torch.optim.lr_scheduler import LambdaLR
    from torch.optim import Adam
    # cfg['lr_start'] = 5e-5
    # cfg['epochs'] = 50
    params_list = model.parameters()
    optimizer = Adam(params_list, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg.epochs) ** 0.9)

    ####################
    # 定义损失函数
    ####################
    ## enet weight
    # weight = [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000]
    ## median_freq_balancing
    weight = [0.0118, 0.2378, 0.7091, 1.0000, 1.9267, 1.5433, 0.9057, 3.2556, 1.0686]
    class_weight =torch.tensor(weight)

    # train_criterion = LovaszSoftmax().to(device)
    train_criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)
    test_criterion = nn.CrossEntropyLoss().to(device)

    ####################
    # 定义指标 包含unlabel，记录训练过程中loss，mPA，mIoU
    ####################
    train_loss_meter = averageMeter()
    test_loss_meter = averageMeter()
    running_metrics_test = runningScore(n_classes=N_CLASSES)
    best_test = 0

    # ------------------------------------------ 以下为训练迭代流程  ----------------------------------------------
    # 每个epoch迭代循环
    for ep in range(cfg['epochs']):

        # training
        model.train()
        train_loss_meter.reset()
        for i, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()  # 梯度清零

            # ------- forward -------
            image = sample['img'].to(device)
            depth = sample['dep'].to(device)
            label = sample['label_cls'].to(device)

            predict = model(image, depth)
            loss = train_criterion(predict, label)

            # ------- forward end -------

            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item())

        scheduler.step(ep)

        # test
        with torch.no_grad():
            model.eval()
            running_metrics_test.reset()
            test_loss_meter.reset()
            for i, sample in enumerate(test_loader):
                image = sample['img'].to(device)
                depth = sample['dep'].to(device)
                label = sample['label_cls'].to(device)
                predict = model(image, depth)

                loss = test_criterion(predict, label)
                test_loss_meter.update(loss.item())

                predict = predict.max(1)[1].cpu().numpy()  # [1, h, w]
                label = label.cpu().numpy()
                running_metrics_test.update(label, predict)

        train_loss = train_loss_meter.avg
        test_loss = test_loss_meter.avg

        test_macc = running_metrics_test.get_scores()[0]["class_acc: "]
        test_miou = running_metrics_test.get_scores()[0]["mIou: "]
        test_avg = (test_macc + test_miou) / 2

        logger.info(f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] loss={train_loss:.3f}/{test_loss:.3f}, mPA={test_macc:.3f}, miou={test_miou:.3f}, avg={test_avg:.3f}')
        if test_avg > best_test:
            best_test = test_avg
            save_ckpt(logdir, model)
        # save_ckpt(logdir, model, prefix='final')

    # ------------------------------------------ 训练结束  ----------------------------------------------


if __name__ == '__main__':

    cfg_path = "configs/cfg_baseline.py"
    run(cfg_path)
