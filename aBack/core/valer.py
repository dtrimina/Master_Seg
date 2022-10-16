import mmcv
import glob

import numpy as np
import torch
import os

from tqdm import tqdm
from pathlib import Path

from aBack.core.trainer import Exp
from utils import SegMeter, FreespaceMeter

from torchvision.transforms import transforms

from data import SegImage

# from pprint import pprint
#
# from toolbox.utils import draw_seg_over_img


class Valer(object):

    def __init__(self, logdir, test_dir=None, front_dist=7, img_size=(384, 384)):
        self.logdir = logdir
        self.cfg = mmcv.Config.fromfile(glob.glob(f'{logdir}/*.py')[0])

        # init model
        self.model_name = self.cfg.model_name
        self.model = Exp.get_model(name=self.model_name, cfg=self.cfg)

        if self.model_name == 'model_pld_and_seg':
            self.model.load_state_dict(torch.load(os.path.join(logdir, 'latest.pth')))
            # self.model.load_state_dict(torch.load(os.path.join(logdir, 'ckpt.pth'))['model'])
            self.model.cuda().eval()
        else:
            state = torch.load(os.path.join(logdir, 'ckpt.pth'))  # fixme
            self.model.load_state_dict(state['model'])
            self.model.cuda().eval()

        self.seg_meter = SegMeter(n_classes=16, ignore_index=255)
        self.fs_meter = FreespaceMeter()

        self.im_to_tensor = transforms.ToTensor()

        self.model_input_size = (self.cfg.image_w, self.cfg.image_h)
        assert self.model_input_size == img_size
        self.testset_dir = test_dir

        self.car_lenth = 4.8  # m
        self.front_dist = front_dist    # m
        self.fResolution = (2 * self.front_dist + self.car_lenth) / self.model_input_size[1]

        print(self.front_dist, self.car_lenth, self.model_input_size, self.fResolution)


    @torch.no_grad()
    def model_inference(self, cv2_img):
        img = self.im_to_tensor(cv2_img)
        img = img.unsqueeze(0).cuda()

        # model inference
        pred_dict = self.model(img)
        return pred_dict

    def get_predict(self, cv2_img, label=None):
        cv2_img = mmcv.imresize(cv2_img, size=self.model_input_size, interpolation='nearest')
        pred_dict = self.model_inference(cv2_img)

        output_dict = {}
        output_dict['cv2_img'] = cv2_img
        output_dict['pred_cls_4dtensor'] = pred_dict['pred_cls'].cpu()

        if 'pred_cls' in pred_dict:
            output_dict['pred_cls'] = pred_dict['pred_cls'].max(1)[1].squeeze(0).cpu().numpy()
            output_dict['pred_cls'][output_dict['pred_cls'] == 0] = 2  # set self car as road

        if 'pred_freespace' in pred_dict:
            pred_freespace = torch.sigmoid(pred_dict['pred_freespace']).squeeze(0).squeeze(0).cpu().numpy()
            pred_freespace = (pred_freespace > 0.5).astype(np.int32)
            output_dict['pred_freespace'] = pred_freespace

        if 'pred_edge' in pred_dict:
            pred_edge = torch.sigmoid(pred_dict['pred_edge']).squeeze(0).squeeze(0).cpu().numpy()
            pred_edge = (pred_edge > 0.5).astype(np.int32)
            output_dict['pred_edge'] = pred_edge

        if label is not None:
            output_dict['label_cls'] = mmcv.imresize(label, size=self.model_input_size, interpolation='nearest')
            output_dict['label_cls_3dtensor'] = torch.from_numpy(output_dict['label_cls']).unsqueeze(0).long()

        return output_dict

    def load_testcase(self, case_dir):

        # df_bad = pd.read_csv(
        #     r'E:\work\pytorch_training_tool_seg\Space_seg\scripts\loss_summary\test_df_bad_top_k.csv',
        #     encoding='utf-8')
        # df_bad = df_bad['img_path'].tolist()

        # testset_infos = []
        # testdir = Path(testset_dir)
        case_infos = []
        mask_gray_dir = case_dir.joinpath('seg/mask_gray')
        assert mask_gray_dir.exists() is True
        for mask_gray_path in mask_gray_dir.iterdir():
            mask_gray_path = str(mask_gray_path)
            # img_path = mask_gray_path.replace('seg/mask_gray', 'image').replace('.png', '.jpg')
            img_path = os.path.join(str(case_dir), 'image', os.path.basename(mask_gray_path).replace('.png', '.jpg'))
            assert os.path.exists(img_path)

            # if img_path in df_bad:
            #     continue

            info = {
                'img_path': img_path,
                'mask_gray_path': mask_gray_path
            }
            case_infos.append(info)

        print(len(case_infos))

        # case_infos = self.trainset.data_infos[:2000]

        return case_infos

    @torch.no_grad()
    def eval_acc(self, write_txt=True, contain_freespace=False):
        assert self.testset_dir is not None
        testset_dir = Path(self.testset_dir)
        num_test_img = 0

        self.seg_meter.contain_freespace = contain_freespace
        save_info = f'test image size is {self.model_input_size}\n\n'

        for case_id, case_dir in enumerate(testset_dir.iterdir()):
            case_infos = self.load_testcase(case_dir)
            num_test_img += len(case_infos)
            self.current_segmeter = SegMeter(n_classes=16, ignore_index=255)
            self.current_fsmeter = FreespaceMeter()

            save_info += f'case-{case_id:2d} ------<<<<< {str(case_dir)},   num_imgs={len(case_infos)}\n'

            for i, sample in enumerate(tqdm(case_infos, total=len(case_infos))):

                segimg = SegImage(img_path=sample['img_path'], mask_gray_path=sample['mask_gray_path'])

                pred_dict = self.get_predict(cv2_img=segimg.img_data(), label=segimg.label_cls())

                pred_cls = pred_dict['pred_cls'][np.newaxis, :, :]
                label_cls = pred_dict['label_cls'][np.newaxis, :, :]

                # # for test set wall -> obstacle
                # pred_cls[pred_cls == 8] = 14
                # label_cls[label_cls == 8] = 14

                self.current_segmeter.update(label_cls, pred_cls)

                if contain_freespace:
                    pred_freespace = pred_dict['pred_freespace'][np.newaxis, :, :]
                    label_freespace = segimg.label_fs_map(size=self.test_label_size)[np.newaxis, :, :]
                    self.current_fsmeter.update(label_freespace, pred_freespace)

            self.seg_meter.update_from_meter(self.current_segmeter)
            self.fs_meter.update_from_meter(self.current_fsmeter)

            # current case result to csv
            save_info += self.format_result(self.current_segmeter.get_scores(), self.current_fsmeter.get_scores())
            save_info += "\n\n"

            # save_path = os.path.join(self.logdir, 'result_back.txt')
            # with open(save_path, 'a') as fp:
            #     fp.writelines(save_result)

        # all case result to csv
        save_info += f'case-overall ------<<<<< num_imgs={num_test_img}\n'
        save_info += self.format_result(self.seg_meter.get_scores(), self.fs_meter.get_scores())

        # confusion matrix
        format_cf_matrix = self.format_confusion_matrix(self.seg_meter.get_confusion_matrix())
        save_info += '\n\n confusion matrix:'
        save_info += format_cf_matrix

        if write_txt:
            save_path = os.path.join(self.logdir, 'result.txt')
            with open(save_path, 'w', encoding='utf-8') as fp:
                fp.write(save_info)

        return

    def eval_acc_384x384_7m(self, write_txt=True, contain_freespace=False):
        assert self.model_input_size == (384, 384)
        assert self.testset_dir is not None and '7m' in os.path.basename(self.testset_dir).lower()

        # load test imgs
        case_infos = []
        image_names = sorted(os.listdir(os.path.join(self.testset_dir, 'image')))
        for name in image_names:
            info = {
                'img_path': os.path.join(self.testset_dir, 'image', name),
                'mask_gray_path': os.path.join(self.testset_dir, 'mask_gray', name.split('.')[0] + '.png')
            }
            case_infos.append(info)
        num_test_img = len(case_infos)

        self.seg_meter.contain_freespace = contain_freespace
        save_info = f'test image size is {self.model_input_size}, fResolution is {self.fResolution}\n\n'

        for i, sample in enumerate(tqdm(case_infos, total=len(case_infos))):
            segimg = SegImage(img_path=sample['img_path'], mask_gray_path=sample['mask_gray_path'])

            if segimg.dist != '7m':
                segimg.centercrop_segimage_from_12m_to_7m(format=False)

            # tmp = segimg.label_cls_over_img(src_rate=0.4)
            # cv2.imshow('img', tmp)
            # cv2.waitKey()

            pred_dict = self.get_predict(cv2_img=segimg.img_data(), label=segimg.label_cls())
            pred_cls = pred_dict['pred_cls'][np.newaxis, :, :]
            label_cls = pred_dict['label_cls'][np.newaxis, :, :]
            self.seg_meter.update(label_cls, pred_cls)

            if contain_freespace:
                pred_freespace = pred_dict['pred_freespace'][np.newaxis, :, :]
                label_freespace = segimg.label_fs_map(size=self.model_input_size)[np.newaxis, :, :]
                self.fs_meter.update(label_freespace, pred_freespace)

        # all case result to csv
        save_info += f'case-overall ------<<<<< num_imgs={num_test_img}\n'
        save_info += self.format_result(self.seg_meter.get_scores(), self.fs_meter.get_scores())

        # confusion matrix
        format_cf_matrix = self.format_confusion_matrix(self.seg_meter.get_confusion_matrix())
        save_info += '\n\n confusion matrix:'
        save_info += format_cf_matrix

        if write_txt:
            save_path = os.path.join(self.logdir, 'result.txt')
            with open(save_path, 'w', encoding='utf-8') as fp:
                fp.write(save_info)

    def eval_acc_576x576_7m_by_flip_padding(self, write_txt=True, contain_freespace=False, area_7m = ()):
        assert self.model_input_size == (576, 576)
        assert self.testset_dir is not None and '7m' in os.path.basename(self.testset_dir).lower()

        # load test imgs
        case_infos = []
        image_names = sorted(os.listdir(os.path.join(self.testset_dir, 'image')))
        for name in image_names:
            info = {
                'img_path': os.path.join(self.testset_dir, 'image', name),
                'mask_gray_path': os.path.join(self.testset_dir, 'mask_gray', name.split('.')[0] + '.png')
            }
            case_infos.append(info)
        num_test_img = len(case_infos)

        self.seg_meter.contain_freespace = contain_freespace
        save_info = f'test image size is {self.model_input_size}, fResolution is {self.fResolution}\n\n'

        for i, sample in enumerate(tqdm(case_infos, total=len(case_infos))):
            segimg = SegImage(img_path=sample['img_path'], mask_gray_path=sample['mask_gray_path'])
            # crop center 7m area
            pred_dict = self.get_predict(cv2_img=segimg.img_data(), label=segimg.label_cls())
            pred_cls = pred_dict['pred_cls'][101:477, 101:477][np.newaxis, :, :]
            label_cls = pred_dict['label_cls'][101:477, 101:477][np.newaxis, :, :]

            self.seg_meter.update(label_cls, pred_cls)

            if contain_freespace:
                pred_freespace = pred_dict['pred_freespace'][np.newaxis, :, :]
                label_freespace = segimg.label_fs_map(size=self.model_input_size)[np.newaxis, :, :]
                self.fs_meter.update(label_freespace, pred_freespace)

        # all case result to csv
        save_info += f'case-overall ------<<<<< num_imgs={num_test_img}\n'
        save_info += self.format_result(self.seg_meter.get_scores(), self.fs_meter.get_scores())

        # confusion matrix
        format_cf_matrix = self.format_confusion_matrix(self.seg_meter.get_confusion_matrix())
        save_info += '\n\n confusion matrix:'
        save_info += format_cf_matrix

        if write_txt:
            save_path = os.path.join(self.logdir, 'result.txt')
            with open(save_path, 'w', encoding='utf-8') as fp:
                fp.write(save_info)

    def eval_acc_576x576_12m(self, dist_range, write_txt=True, contain_freespace=False):
        assert self.model_input_size == (576, 576)
        assert self.testset_dir is not None and '12m' in os.path.basename(self.testset_dir).lower()

        self.seg_meter.set_roi(dist_limit=dist_range)

        # load test imgs
        case_infos = []
        image_names = sorted(os.listdir(os.path.join(self.testset_dir, 'image')))
        for name in image_names:
            info = {
                'img_path': os.path.join(self.testset_dir, 'image', name),
                'mask_gray_path': os.path.join(self.testset_dir, 'mask_gray', name.split('.')[0] + '.png')
            }
            case_infos.append(info)
        num_test_img = len(case_infos)

        self.seg_meter.contain_freespace = contain_freespace
        save_info = f'test image size is {self.model_input_size}, fResolution is {self.fResolution}\n\n'

        for i, sample in enumerate(tqdm(case_infos, total=len(case_infos))):
            segimg = SegImage(img_path=sample['img_path'], mask_gray_path=sample['mask_gray_path'])

            pred_dict = self.get_predict(cv2_img=segimg.img_data(), label=segimg.label_cls())
            pred_cls = pred_dict['pred_cls'][np.newaxis, :, :]
            label_cls = pred_dict['label_cls'][np.newaxis, :, :]
            self.seg_meter.update(label_cls, pred_cls)

            if contain_freespace:
                pred_freespace = pred_dict['pred_freespace'][np.newaxis, :, :]
                label_freespace = segimg.label_fs_map(size=self.model_input_size)[np.newaxis, :, :]
                self.fs_meter.update(label_freespace, pred_freespace)

        # all case result to csv
        save_info += f'case-overall ------<<<<< num_imgs={num_test_img}\n'
        save_info += self.format_result(self.seg_meter.get_scores(), self.fs_meter.get_scores())

        # confusion matrix
        format_cf_matrix = self.format_confusion_matrix(self.seg_meter.get_confusion_matrix())
        save_info += '\n\n confusion matrix:'
        save_info += format_cf_matrix

        if write_txt:
            save_path = os.path.join(self.logdir, 'result.txt')
            with open(save_path, 'w', encoding='utf-8') as fp:
                fp.write(save_info)

    def eval_acc_576x576_4m(self, dist_range, write_txt=True, contain_freespace=False):
        assert self.model_input_size == (576, 576)
        assert self.testset_dir is not None and '12m' in os.path.basename(self.testset_dir).lower()

        self.seg_meter.set_roi(dist_limit=dist_range)

        # load test imgs
        case_infos = []
        image_names = sorted(os.listdir(os.path.join(self.testset_dir, 'image')))
        for name in image_names:
            info = {
                'img_path': os.path.join(self.testset_dir, 'image', name),
                'mask_gray_path': os.path.join(self.testset_dir, 'mask_gray', name.split('.')[0] + '.png')
            }
            case_infos.append(info)
        num_test_img = len(case_infos)

        self.seg_meter.contain_freespace = contain_freespace
        save_info = f'test image size is {self.model_input_size}, fResolution is {self.fResolution}\n\n'

        for i, sample in enumerate(tqdm(case_infos, total=len(case_infos))):
            segimg = SegImage(img_path=sample['img_path'], mask_gray_path=sample['mask_gray_path'])

            segimg.inplace_centercrop_segimage(crop_dist=5)
            #
            # print(self.fResolution)
            # cv2.imshow('img', segimg.label_cls_over_img(src_rate=0.4))
            # cv2.imshow('img_pred', segimg.pred_cls_over_img(src_rate=0.4))
            # cv2.waitKey()

            pred_dict = self.get_predict(cv2_img=segimg.img_data(), label=segimg.label_cls())
            pred_cls = pred_dict['pred_cls'][np.newaxis, :, :]
            label_cls = pred_dict['label_cls'][np.newaxis, :, :]
            self.seg_meter.update(label_cls, pred_cls)

            if contain_freespace:
                pred_freespace = pred_dict['pred_freespace'][np.newaxis, :, :]
                label_freespace = segimg.label_fs_map(size=self.model_input_size)[np.newaxis, :, :]
                self.fs_meter.update(label_freespace, pred_freespace)

        # all case result to csv
        save_info += f'case-overall ------<<<<< num_imgs={num_test_img}\n'
        save_info += self.format_result(self.seg_meter.get_scores(), self.fs_meter.get_scores())

        # confusion matrix
        format_cf_matrix = self.format_confusion_matrix(self.seg_meter.get_confusion_matrix())
        save_info += '\n\n confusion matrix:'
        save_info += format_cf_matrix

        if write_txt:
            save_path = os.path.join(self.logdir, 'result.txt')
            with open(save_path, 'w', encoding='utf-8') as fp:
                fp.write(save_info)

    def eval_acc_576x576_xm(self, write_txt=True, contain_freespace=False):
        pass

    def format_result(self, results, fs_results=None):
        CLASSES = ('none', 'car', 'road', 'ped', 'zebra(nan)', 'parkline(nan)', 'arrow(nan)', 'curbstone', 'wallcolumn',
                   #    9,          10,           11,            12,              13,            14             15
                   'speed bump', 'lane line(nan)', 'park lock', 'vehicle stoper', 'traffic cone', 'obstacle',
                   'park area(nan)')

        columns = ("metric",) + ('mean',) + CLASSES
        index = ['ratio(%)', 'PA', 'mPA', 'mIoU', 'mBA', 'mFSBA']

        max_column_elem_size = max([len(i) for i in columns])

        ratio_rst = [index[0], ] + [f'100', ] + [f'{i * 100:.2f}' for i in results['pixel_ratio_per_class']]
        pa_all = [index[1], ] + [f"{results['PA']:.4f}", ]
        pa_rst = [index[2], ] + [f"{results['mPA']:.4f}", ] + [f'{i:.4f}' for i in results['pa_per_cls']]
        iou_rst = [index[3], ] + [f"{results['mIoU']:.4f}", ] + [f'{i:.4f}' for i in results['iou_per_cls']]
        ba_rst = [index[4], ] + [f"{results['mBA']:.4f}", ] + [f'{i:.4f}' for i in results['ba_per_radiu']]
        fs_ba_rst = [index[5], ] + [f"{results['mFSBA']:.4f}", ] + [f'{i:.4f}' for i in results['fs_ba_per_radiu']]

        rst_matrix = [columns, ratio_rst, pa_all, pa_rst, iou_rst, ba_rst, fs_ba_rst]
        info = ""
        for i in range(len(rst_matrix)):
            info += ','.join([i.center(max_column_elem_size) for i in rst_matrix[i]]) + '\n'

        if fs_results is not None:
            fs_pa_all = ['fs_PA', f"{fs_results['PA']:.4f}"]
            fs_pa_rst = ['fs_mPA', ] + [f"{fs_results['mPA']:.4f}", ] + [f'{i:.4f}' for i in fs_results['pa_per_cls']]
            fs_fs_ba_rst = ['fs_BA', ] + [f"{fs_results['mFSBA']:.4f}", ] + [f'{i:.4f}' for i in
                                                                             fs_results['fs_ba_per_radiu']]
            rst_matrix = [fs_pa_all, fs_pa_rst, fs_fs_ba_rst]
            info += "\nfreespace_map_rst\n"
            for i in range(len(rst_matrix)):
                info += ','.join([i.center(max_column_elem_size) for i in rst_matrix[i]]) + '\n'

        print(info)

        return info

    def format_confusion_matrix(self, confusion_matrix):
        info = ""
        for line in confusion_matrix:
            line_info = ",".join([f'{int(i):7d}' for i in line])
            info += line_info
            info += '\n'
        print(info)
        return info
