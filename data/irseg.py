import os
import random

import numpy as np
import pandas as pd

from pathlib import Path
from loguru import logger
import cv2
import common

import torch
import torch.utils.data as data
from torchvision import transforms

from data.augmentations import Resize, PhotoMetricDistortion, RandomFlip, RandomCrop, RandomRotate, \
    GaussianBlur, Pad, Compose
from data.image import SegImage


class IRSeg(data.Dataset):

    def __init__(self, cfg, mode='train'):
        assert mode in ['train', 'test']

        self.mode = mode
        self.cfg = cfg
        self.n_classes = cfg.n_classes
        self.boundary_kernel_size = 5

        # pre-processing
        self.im_to_tensor = transforms.ToTensor()

        target_size = (cfg.image_h, cfg.image_w)
        self.img_aug = Compose([
            PhotoMetricDistortion(brightness_delta_range=(-32, 32), contrast_range=(0.75, 1.25),
                                  saturation_range=(0.75, 1.25), hue_delta=18, prob=0.95),
            GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.2), prob=0.5, kernel_size_options=(3, 5)),
        ])
        self.shape_aug = Compose([
            RandomFlip(prob=0.5, direction="horizontal"),
            Resize(img_scale=target_size, ratio_range=(0.5, 1.5)),
            RandomRotate(prob=0.3, degree=20, pad_val=0, seg_pad_val=0),
            RandomCrop(crop_size=target_size, cat_max_ratio=0.75),
            Pad(size=target_size, pad_val=0, seg_pad_val=0),
        ])

        logger.info(f'set train_img_size={target_size}')
        self.format_resize = Resize(target_size)

        # preload
        self.data_infos = []

        if self.mode == 'train':
            txt_path = os.path.join(cfg.data_dir, 'trainval.txt')
        else:
            txt_path = os.path.join(cfg.data_dir, 'test.txt')

        with open(txt_path, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            name = line.strip()
            info = {
                'img_path': os.path.join(cfg.data_dir, 'seperated_images', f'{name}_rgb.png'),
                'dep_path': os.path.join(cfg.data_dir, 'seperated_images', f'{name}_th.png'),
                'mask_gray_path': os.path.join(cfg.data_dir, 'labels', f'{name}.png')
            }
            self.data_infos.append(info)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):

        info = self.data_infos[index]

        segimg = SegImage(img_path=info['img_path'], dep_path=info['dep_path'], mask_gray_path=info['mask_gray_path'])

        img_h, img_w = segimg.img_data().shape[:2]

        sample = {
            'center_point': (img_w // 2, img_h // 2),
            'img': segimg.img_data(),
            'dep': segimg.dep_data(),
            'label_cls': segimg.label_cls(),
            'gt_semantic_seg': segimg.label_cls().copy(),
        }

        # augmentation
        if self.mode == 'train':
            sample = self.img_aug(sample)
            if common.torch_rand(0, 1) < 0.95:
                sample = self.shape_aug(sample)
            else:
                sample = self.format_resize(sample)
        else:
            sample = self.format_resize(sample)

        # reset segimg
        segimg.set_img_data(sample['img'])
        segimg.set_dep_data(sample['dep'])
        segimg.set_label_cls_data(sample['label_cls'])

        # get contain obj tensor encode
        contain_classes = set(sample['label_cls'].reshape(-1).tolist())
        obj_encode = np.zeros(self.n_classes)
        for cls_id in contain_classes:
            if cls_id == 255:
                continue
            obj_encode[cls_id] = 1

        out = {
            'img': self.im_to_tensor(sample['img']),
            'dep': self.im_to_tensor(sample['dep']),
            'label_cls': torch.from_numpy(np.asarray(sample['label_cls'], dtype=np.int64)).long(),
            'label_saliency': torch.from_numpy(np.asarray(segimg.label_saliency_map(), dtype=np.int64)).long(),
            'label_boundary': torch.from_numpy(np.asarray(segimg.label_boundary_map(kernel_size=self.boundary_kernel_size), dtype=np.int64)).long(),
            'img_name': os.path.basename(info['img_path']),
            # 'label_cls_encode': obj_encode,
            'center_point': sample['center_point']
        }

        return out


if __name__ == '__main__':
    import mmcv

    path = r'../configs/cfg_baseline.py'
    cfg = mmcv.Config.fromfile(path)

    # cfg.data_dir = '/home/workstation/Desktop/SegProject/database/irseg'
    cfg.data_dir = '../../database/irseg'

    dataset = IRSeg(cfg, mode='train')

    print(len(dataset))

    for i in range(len(dataset)):

        index = i
        # index = random.randint(0, len(dataset))
        sample = dataset[index]

        image = sample['img']
        image = (image.numpy().transpose(1, 2, 0) * 255.).astype(np.uint8)
        depth = sample['dep']
        depth = (depth.numpy().transpose(1, 2, 0) * 255.).astype(np.uint8)
        label_cls = sample['label_cls'].numpy()

        label_saliency = sample['label_saliency'].numpy().astype(np.uint8) * 255
        label_boundary = sample['label_boundary'].numpy().astype(np.uint8) * 255

        segimg = SegImage(img_data=image, dep_data=depth, label_cls_data=label_cls)
        labeled_data = segimg.label_cls_over_img_with_all_info(src_rate=0.4).copy()

        cv2.imshow('img', labeled_data)
        cv2.imshow('boundary', label_boundary)

        cv2.imshow('saliency', label_saliency)
        cv2.waitKey()
