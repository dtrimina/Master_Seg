import numpy as np
import cv2
import mmcv
from data import mask_gray2freespace, freespace_to_fs_boundary


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DictAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum_loss_dict = {}

    def update(self, loss_dict):
        for k, v in loss_dict.items():
            if k not in self.sum_loss_dict:
                self.sum_loss_dict[k] = 0
            self.sum_loss_dict[k] += v.item()
        self.count += 1

    def get_avg(self, key):
        assert key in self.sum_loss_dict
        return self.sum_loss_dict[key] / self.count

    def info(self):
        prefix = ''
        for k, v in self.sum_loss_dict.items():
            prefix += f'{k}: {v / self.count:.5f}, '
        return prefix


class SegMeter(object):
    '''
        n_classes: database的类别,包括背景
        ignore_index: 需要忽略的类别id,一般为未标注id, eg. CamVid.id_unlabel
    '''

    def __init__(self, n_classes, ignore_index=None):
        self.n_classes = n_classes

        if ignore_index is None or ignore_index < 0 or ignore_index > n_classes:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

        self.mBA_boundary_kernels = [3, 5, 7, 9, 11]
        self.mFSBA_boundary_kernels = [3, 5, 7, 9, 11]
        self.fs_ids = [2, 4, 5, 6, 9, 10, 12, 15]

        self.roi_map_15m_576x576 = None
        self.roi_map_box_list = None  # [(outbox_p1, outbox_p2), (innerbox_p1, innerbox_P2)]

        self.reset()

    def set_roi(self, dist_limit):
        assert isinstance(dist_limit, tuple) and len(dist_limit) == 2
        assert dist_limit[1] <= 12
        print('set roi option only support for 12m 576x576 img')

        roi_map = np.zeros((576, 576))
        pixel_per_m = 576 / (12 + 12 + 4.8)
        # pixel_per_m = 576 / (5 + 5 + 4.8)

        max_dist_shift = pixel_per_m * (12 - dist_limit[1])
        min_dist_shift = pixel_per_m * (12 - dist_limit[0])
        # max_dist_shift = pixel_per_m * (5 - dist_limit[1])
        # min_dist_shift = pixel_per_m * (5 - dist_limit[0])

        max_dist_shift = int(max_dist_shift)
        min_dist_shift = int(min_dist_shift)

        roi_map[max_dist_shift: 576 - max_dist_shift, max_dist_shift: 576 - max_dist_shift] = 1
        self.roi_map_box_list = [((max_dist_shift, max_dist_shift), (576 - max_dist_shift, 576 - max_dist_shift))]
        if dist_limit[0] != 0:
            roi_map[min_dist_shift: 576 - min_dist_shift, min_dist_shift: 576 - min_dist_shift] = 0
            self.roi_map_box_list.append(((min_dist_shift, min_dist_shift), (576 - min_dist_shift, 576 - min_dist_shift)))

        self.roi_map_15m_576x576 = roi_map

        # cv2.imshow('roi', self.roi_map_15m_576x576)
        # cv2.waitKey()

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

        self.mBA_boundary_all_pixel_num = np.zeros(shape=(len(self.mBA_boundary_kernels, )))
        self.mBA_boundary_acc_pixel_num = np.zeros(shape=(len(self.mBA_boundary_kernels, )))

        self.num_pixel_per_class = np.zeros((self.n_classes,))
        self.mFSBA_boundary_all_pixel_num = np.zeros(shape=(len(self.mFSBA_boundary_kernels, )))
        self.mFSBA_boundary_acc_pixel_num = np.zeros(shape=(len(self.mFSBA_boundary_kernels, )))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        if self.roi_map_15m_576x576 is not None:
            mask = mask & (self.roi_map_15m_576x576 == 1).flatten()

        if self.ignore_index:
            for idx in self.ignore_index:
                mask &= (label_true != idx)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def _compute_boundary_acc_multi_class(self, label_true, label_pred):
        h, w = label_true.shape
        classes = np.unique(label_true)

        for i, ks in enumerate(self.mBA_boundary_kernels):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
            boundary_region = np.zeros_like(label_true)

            # make gt boundary
            for c in classes:
                # Skip ignore index
                if c == self.ignore_index:
                    continue

                gt_class = (label_true == c).astype(np.uint8)
                class_bound = cv2.morphologyEx(gt_class, cv2.MORPH_GRADIENT, kernel)
                boundary_region += class_bound

            boundary_region = boundary_region > 0
            boundary_region = boundary_region & (label_true != self.ignore_index)  # filter ignore area

            if self.roi_map_15m_576x576 is not None:
                boundary_region = boundary_region & (self.roi_map_15m_576x576 == 1)

            label_true_in_bound = label_true[boundary_region]
            label_pred_in_bound = label_pred[boundary_region]

            num_all_edge_pixel = boundary_region.sum()
            num_acc_edge_pixel = (label_true_in_bound == label_pred_in_bound).sum()

            self.mBA_boundary_all_pixel_num[i] += num_all_edge_pixel
            self.mBA_boundary_acc_pixel_num[i] += num_acc_edge_pixel

    def _compute_fs_boundary_acc(self, label_true, label_pred):
        fs_true = mask_gray2freespace(label_true)
        fs_pred = mask_gray2freespace(label_pred)

        for i, ks in enumerate(self.mFSBA_boundary_kernels):
            boundary_region = freespace_to_fs_boundary(fs_true, kernel_size=ks) == 1

            if self.roi_map_15m_576x576 is not None:
                boundary_region = boundary_region & (self.roi_map_15m_576x576 == 1)

            # cv2.imshow('fs_true', (fs_true * 255.).astype(np.uint8))
            # cv2.imshow('fs_pred', (fs_pred * 255).astype(np.uint8))
            # cv2.imshow('gt_bound', (boundary_region* 255).astype(np.uint8))
            # cv2.waitKey()

            fs_true_in_bound = fs_true[boundary_region]
            fs_pred_in_bound = fs_pred[boundary_region]

            num_all_edge_pixel = boundary_region.sum()
            num_acc_edge_pixel = (fs_true_in_bound == fs_pred_in_bound).sum()

            self.mFSBA_boundary_all_pixel_num[i] += num_all_edge_pixel
            self.mFSBA_boundary_acc_pixel_num[i] += num_acc_edge_pixel

    def _gt_pixel_summary(self, lt):
        for c in range(self.n_classes):
            self.num_pixel_per_class[c] += (lt == c).sum()

    def update(self, label_trues, label_preds, ignore_fs_boundary=False):
        assert len(label_trues.shape) == 3, f'inps shape should be (num_img, h, w)'

        if self.roi_map_15m_576x576 is not None:
            assert label_trues.shape[1] == label_trues.shape[2] and label_trues.shape[2] == 576

        for lt, lp in zip(label_trues, label_preds):

            # lt = mmcv.imresize(lt, size=self.val_img_size, interpolation='nearest')
            # lp = mmcv.imresize(lp, size=self.val_img_size, interpolation='nearest')

            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

            self._compute_boundary_acc_multi_class(lt, lp)

            self._gt_pixel_summary(lt)

            if not ignore_fs_boundary:
                self._compute_fs_boundary_acc(lt, lp)

    def update_from_meter(self, other_meter):
        self.confusion_matrix += other_meter.confusion_matrix
        self.mBA_boundary_all_pixel_num += other_meter.mBA_boundary_all_pixel_num
        self.mBA_boundary_acc_pixel_num += other_meter.mBA_boundary_acc_pixel_num
        self.mFSBA_boundary_all_pixel_num += other_meter.mFSBA_boundary_all_pixel_num
        self.mFSBA_boundary_acc_pixel_num += other_meter.mFSBA_boundary_acc_pixel_num
        self.num_pixel_per_class += other_meter.num_pixel_per_class

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - pixel_acc:
            - class_acc: class mean acc
            - mIou :     mean intersection over union
            - fwIou:     frequency weighted intersection union
        """

        hist = self.confusion_matrix

        # ignore unlabel
        if self.ignore_index is not None:
            for index in self.ignore_index:
                hist = np.delete(hist, index, axis=0)
                hist = np.delete(hist, index, axis=1)

        pixel_acc = np.diag(hist).sum() / hist.sum()
        pixel_acc_per_cls = np.diag(hist) / hist.sum(axis=1)
        mPA = np.nanmean(pixel_acc_per_cls)
        iou_per_cls = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mIoU = np.nanmean(iou_per_cls)
        freq = hist.sum(axis=1) / hist.sum()
        fw_iou = (freq[freq > 0] * iou_per_cls[freq > 0]).sum()

        # set unlabel as nan
        if self.ignore_index is not None:
            for index in self.ignore_index:
                iu = np.insert(iou_per_cls, index, np.nan)

        # cls_iu = dict(zip(range(self.n_classes), iu))
        # cls_acc = dict(zip(range(self.n_classes), cls_acc))

        # compute mBA
        pixel_acc_per_radiu = self.mBA_boundary_acc_pixel_num / self.mBA_boundary_all_pixel_num
        mBA = pixel_acc_per_radiu.mean()

        # compute mFSBA
        fs_pixel_acc_per_radiu = self.mFSBA_boundary_acc_pixel_num / self.mFSBA_boundary_all_pixel_num
        mFSBA = fs_pixel_acc_per_radiu.mean()

        # gt num_pixel per cls summary
        pixel_ratio_per_class = self.num_pixel_per_class / self.num_pixel_per_class.sum()

        out = {
            'PA': pixel_acc,
            'mPA': mPA,
            'mIoU': mIoU,
            'mBA': mBA,
            'mFSBA': mFSBA,

            'pa_per_cls': pixel_acc_per_cls,
            'iou_per_cls': iou_per_cls,
            'mBA_radius': self.mBA_boundary_kernels,
            'ba_per_radiu': pixel_acc_per_radiu,
            'mFSBA_radius': self.mFSBA_boundary_kernels,
            'fs_ba_per_radiu': fs_pixel_acc_per_radiu,

            'pixel_ratio_per_class': pixel_ratio_per_class
        }
        return out

    def get_confusion_matrix(self):
        return self.confusion_matrix


class FreespaceMeter(object):
    '''
        n_classes: database的类别,包括背景
        ignore_index: 需要忽略的类别id,一般为未标注id, eg. CamVid.id_unlabel
    '''

    def __init__(self, val_img_size=(768, 768)):
        self.valid = False
        self.val_img_size = val_img_size
        self.mFSBA_boundary_kernels = [9]

        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((2, 2))

        self.mFSBA_boundary_all_pixel_num = np.zeros(shape=(len(self.mFSBA_boundary_kernels, )))
        self.mFSBA_boundary_acc_pixel_num = np.zeros(shape=(len(self.mFSBA_boundary_kernels, )))

    def _fast_hist(self, label_true, label_pred, n_class=2):
        hist = np.bincount(
            n_class * label_true.astype(int) + label_pred, minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def _compute_fs_boundary(self, fs_true, fs_pred):
        for i, ks in enumerate(self.mFSBA_boundary_kernels):
            boundary_region = freespace_to_fs_boundary(fs_true, kernel_size=ks) == 1

            # cv2.imshow('fs_true', (fs_true * 255.).astype(np.uint8))
            # cv2.imshow('fs_pred', (fs_pred * 255).astype(np.uint8))
            # cv2.imshow('gt_bound', (boundary_region * 255).astype(np.uint8))
            # cv2.waitKey()

            fs_true_in_bound = fs_true[boundary_region]
            fs_pred_in_bound = fs_pred[boundary_region]

            num_all_edge_pixel = boundary_region.sum()
            num_acc_edge_pixel = (fs_true_in_bound == fs_pred_in_bound).sum()

            self.mFSBA_boundary_all_pixel_num[i] += num_all_edge_pixel
            self.mFSBA_boundary_acc_pixel_num[i] += num_acc_edge_pixel

    def update(self, label_trues, label_preds):
        if self.valid is False:
            self.valid = True

        assert len(label_trues.shape) == 3, f'inps shape should be (num_img, h, w)'
        for lt, lp in zip(label_trues, label_preds):
            lt = mmcv.imresize(lt, size=self.val_img_size, interpolation='nearest')
            lp = mmcv.imresize(lp, size=self.val_img_size, interpolation='nearest')

            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
            self._compute_fs_boundary(lt, lp)

    def update_from_meter(self, other_meter):
        self.valid = other_meter.valid
        self.confusion_matrix += other_meter.confusion_matrix
        self.mFSBA_boundary_all_pixel_num += other_meter.mFSBA_boundary_all_pixel_num
        self.mFSBA_boundary_acc_pixel_num += other_meter.mFSBA_boundary_acc_pixel_num

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - pixel_acc:
            - class_acc: class mean acc
            - mIou :     mean intersection over union
            - fwIou:     frequency weighted intersection union
        """

        if self.valid is False:
            return None

        hist = self.confusion_matrix

        pixel_acc = np.diag(hist).sum() / hist.sum()
        pixel_acc_per_cls = np.diag(hist) / hist.sum(axis=1)
        mPA = np.nanmean(pixel_acc_per_cls)

        # compute mFSBA
        fs_pixel_acc_per_radiu = self.mFSBA_boundary_acc_pixel_num / self.mFSBA_boundary_all_pixel_num
        mFSBA = fs_pixel_acc_per_radiu.mean()

        out = {
            'PA': pixel_acc,
            'mPA': mPA,
            'mFSBA': mFSBA,

            'pa_per_cls': pixel_acc_per_cls,
            'mFSBA_radius': self.mFSBA_boundary_kernels,
            'fs_ba_per_radiu': fs_pixel_acc_per_radiu,
        }

        return out

    def get_confusion_matrix(self):
        return self.confusion_matrix
