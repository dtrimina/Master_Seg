#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os

import cv2
import numpy as np
from aBack.core.valer import Valer
from pathlib import Path

from data.image import SegImage
from utils import SegMeter

# from toolbox.data.dataset import SegTrainSet
# from toolbox.utils import draw_seg_over_img

# SEG_DIR = 'vis/results/seg'


class SegVisualizer(object):

    def __init__(self, logdir):
        self.img_size = (576, 576)

        self.valer = Valer(logdir=logdir, img_size=self.img_size)
        self.seg_meter = SegMeter(n_classes=16, ignore_index=255)

    def inference(self, test_case_dir):

        # case_infos = self.valer.load_testcase(test_case_dir)

        case_infos = []
        image_names = sorted(os.listdir(os.path.join(test_case_dir, 'image')))
        for name in image_names:
            info = {
                'img_path': os.path.join(test_case_dir, 'image', name),
                'mask_gray_path': os.path.join(test_case_dir, 'mask_gray', name.split('.')[0] + '.png')
            }
            case_infos.append(info)

        index = 0
        while index >= 0:
            self.seg_meter.reset()
            self.seg_meter.set_roi(dist_limit=(0, 4))
            img_path = case_infos[index]['img_path']
            gray_path = case_infos[index]['mask_gray_path']

            segimg = SegImage(img_path=img_path, mask_gray_path=gray_path, format_size=self.img_size)

            output_dict = self.valer.get_predict(segimg.img_data(), label=segimg.label_cls())

            # get score
            pred_cls = output_dict['pred_cls'][np.newaxis, :, :]
            label_cls = output_dict['label_cls'][np.newaxis, :, :]
            self.seg_meter.update(label_cls, pred_cls)
            results = self.seg_meter.get_scores()

            print(f'==========> current index {index} <==============')
            print(f"==== image_path: {case_infos[index]['img_path']}")
            print(f"==== PA: {results['PA']}, FSBA_10cm: {results['fs_ba_per_radiu'][1]}")

            #
            segimg.set_pred_cls_data(output_dict['pred_cls'])

            # debug imgs
            show_img_gt = segimg.label_cls_over_img(src_rate=0.4, roi_box_list=self.seg_meter.roi_map_box_list)
            show_img_pred = segimg.pred_cls_over_img(src_rate=0.4, roi_box_list=self.seg_meter.roi_map_box_list)
            show_img_fp = segimg.fp_map_over_img(src_rate=0.4, roi_box_list=self.seg_meter.roi_map_box_list)
            show_img_error = segimg.error_map_over_img(src_rate=0.4, roi_box_list=self.seg_meter.roi_map_box_list)
            show_img_gt_fs_boundary = segimg.label_fs_boundary_map(kernel_size=5)
            show_img_pred_fs_boundary = segimg.pred_fs_boundary_map(kernel_size=5)

            # concate pred img
            show_img1 = np.concatenate([show_img_gt, show_img_pred], axis=1)
            show_img2 = np.concatenate([show_img_fp, show_img_error], axis=1)
            show_img = np.concatenate([show_img1, show_img2], axis=0)
            show_img = cv2.resize(show_img, dsize=(960, 960))

            bound = np.concatenate([show_img_gt_fs_boundary, show_img_pred_fs_boundary], axis=1) * 255
            bound = cv2.resize(bound.astype(np.uint8), dsize=(960, 480))

            cv2.imshow('img', show_img)
            cv2.imshow('bound', bound)

            key = cv2.waitKeyEx()

            if index == 0:
                cv2.waitKey()

            if key == 13:  # press enter
                pass

            elif key == 97:  # press left
                if index >= 1:
                    index -= 1

            elif key == 100:  # press any other keys
                if index < len(case_infos) - 1:
                    index += 1

            elif key == 32:
                cv2.waitKey()

            else:
                if index < len(case_infos) - 1:
                    index += 1


if __name__ == '__main__':
    logdir = r'E:\work\Master_Seg\Space_seg\result\Seg_model_baseline_v2_2022-06-21-14-47_576x576'

    visualizer = SegVisualizer(logdir)

    test_dir = Path(r'E:\Dataset\SegData\Testset12M')
    visualizer.inference(test_case_dir=test_dir)
