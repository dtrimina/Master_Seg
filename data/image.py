import numpy as np
import mmcv
import cv2

from common import PALETTE, N_CLASSES


class Draw(object):
    def __init__(self):
        pass

    @staticmethod
    def draw_seg_over_img(img, seg, n_classes, ignore_index=255, src_rate=0.4,):
        img = img.copy()
        row, col = seg.shape[0:2]
        colors = np.zeros((row, col, 3), dtype=np.uint8)

        for i in range(n_classes):
            colors[seg == i] = PALETTE[i]
        colors[seg == ignore_index] = img[seg == ignore_index]
        img = img * src_rate + colors * (1 - src_rate)
        img = img.astype(np.uint8)
        return img

def mask_gray2saliencymap(mask_gray, center_point=None):
    h, w = mask_gray.shape[:2]

    # 16 class to freespace
    binary_mask = np.zeros((h, w, 1), dtype=np.uint8)
    for fs_id in FS_IDS:
        binary_mask[mask_gray == fs_id] = 1

    freespace = find_max_region(binary_mask, center_point=center_point)
    return freespace


def freespace_to_fs_boundary(freespace, kernel_size=3):
    assert kernel_size % 2 == 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    boundary_region = cv2.morphologyEx(freespace, cv2.MORPH_GRADIENT, kernel) > 0
    return boundary_region.astype(np.int32)


class SegImage(object):

    def __init__(self,
                 img_path=None,
                 dep_path=None,
                 mask_gray_path=None,
                 img_data=None,
                 dep_data=None,
                 label_cls_data=None,
                 pred_cls_data=None,
                 format_size=(640, 480)):

        self._img_data = None
        self._dep_data = None
        self._pred_cls_data = None
        self._label_cls_data = None
        self.img_size = format_size
        self.n_classes = 16

        if img_path is not None:
            assert img_data is None
            cv2_img = mmcv.imread(img_path, flag='color', channel_order='bgr', backend='pillow')
            self.set_img_data(cv2_img)

        if dep_path is not None:
            assert dep_data is None
            tmp_dep_data = mmcv.imread(dep_path)
            self.set_dep_data(tmp_dep_data)

        if mask_gray_path is not None:
            assert label_cls_data is None
            label_cls = mmcv.imread(mask_gray_path, flag='unchanged', backend='pillow')
            self.set_label_cls_data(label_cls)

        if img_data is not None:
            assert img_path is None
            self.set_img_data(img_data)

        if dep_data is not None:
            assert dep_path is None
            self.set_dep_data(dep_data)

        if label_cls_data is not None:
            assert mask_gray_path is None
            self.set_label_cls_data(label_cls_data)

        if pred_cls_data is not None:
            self.set_pred_cls_data(pred_cls_data)

    def set_img_data(self, img_data):
        self._img_data = mmcv.imresize(img_data, size=self.img_size)

    def set_dep_data(self, dep_data):
        self._dep_data = mmcv.imresize(dep_data, size=self.img_size)

    def set_pred_cls_data(self, pred_cls_data):
        self._pred_cls_data = mmcv.imresize(pred_cls_data, size=self.img_size, interpolation='nearest')

    def set_label_cls_data(self, label_cls_data):
        self._label_cls_data = mmcv.imresize(label_cls_data, size=self.img_size, interpolation='nearest')

    def img_data(self, size=None):
        assert self._img_data is not None
        if size is None:
            size = self.img_size
        return mmcv.imresize(self._img_data, size=size)

    def dep_data(self, size=None):
        assert self._dep_data is not None
        if size is None:
            size = self.img_size
        return mmcv.imresize(self._dep_data, size=size)

    def label_cls(self, size=None):
        assert self._label_cls_data is not None
        if size is None:
            size = self.img_size
        return mmcv.imresize(self._label_cls_data, size=self.img_size, interpolation='nearest')

    # def pred_cls(self, size=None):  # pred 16 class mask
    #     assert self._pred_cls_data is not None
    #     if size is not None:
    #         assert isinstance(size, tuple) and len(size) == 2
    #         data = mmcv.imresize(self._pred_cls_data, size=self.img_size, interpolation='nearest')
    #     else:
    #         data = self._pred_cls_data
    #     return data
    #
    # def fp_map(self, size=None):  #
    #     assert self._pred_cls_data is not None
    #     assert self._label_cls_data is not None
    #     assert self._pred_cls_data.shape == self._label_cls_data.shape
    #
    #     result = np.zeros(shape=self.img_size, dtype=np.uint8)
    #     for i in range(self.n_classes):
    #         mask0 = np.zeros_like(self._label_cls_data, dtype=np.uint8)
    #         mask1 = np.zeros_like(self._label_cls_data, dtype=np.uint8)
    #         mask0[self._label_cls_data[:] == i] += 1
    #         mask0[self._pred_cls_data[:] == i] += 1
    #         mask1[self._label_cls_data[:] == i] += 1
    #         result[mask0[:] == 1] = i
    #         result[mask1[:] == 1] = 0
    #
    #     if size is not None:
    #         assert isinstance(size, tuple) and len(size) == 2
    #         data = mmcv.imresize(result, size=self.img_size, interpolation='nearest')
    #     else:
    #         data = result
    #     return data

    # def error_map(self, size=None):  # gt �� pred ֮��� error map
    #     assert self._pred_cls_data is not None
    #     assert self._label_cls_data is not None
    #     assert self._pred_cls_data.shape == self._label_cls_data.shape
    #
    #     result = np.ones(shape=self.img_size, dtype=np.uint8) * 255
    #     result[self._label_cls_data != self._pred_cls_data] = self._pred_cls_data[
    #         self._label_cls_data != self._pred_cls_data]
    #
    #     if size is not None:
    #         assert isinstance(size, tuple) and len(size) == 2
    #         data = mmcv.imresize(result, size=self.img_size, interpolation='nearest')
    #     else:
    #         data = result
    #     return data
    #
    # def pred_fs_map(self, size=None):
    #     assert self._pred_cls_data is not None
    #
    #     pred_fs = mask_gray2freespace(self._pred_cls_data)
    #
    #     if size is not None:
    #         assert isinstance(size, tuple) and len(size) == 2
    #         data = mmcv.imresize(pred_fs, size=self.img_size, interpolation='nearest')
    #     else:
    #         data = pred_fs
    #     return data
    #
    # def pred_fs_boundary_map(self, size=None, kernel_size=3):
    #     freespace = self.pred_fs_map(size=size)
    #     fs_boundary_map = freespace_to_fs_boundary(freespace, kernel_size=kernel_size)
    #     return fs_boundary_map
    #
    def label_fs_map(self, size=None, center_point=None):
        assert self._label_cls_data is not None

        label_fs = mask_gray2freespace(self._label_cls_data, center_point)

        if size is not None:
            assert isinstance(size, tuple) and len(size) == 2
            data = mmcv.imresize(label_fs, size=self.img_size, interpolation='nearest')
        else:
            data = label_fs
        return data

    # def label_fs_boundary_map(self, size=None, kernel_size=3, center_point=None):
    #     freespace = self.label_fs_map(size=size, center_point=center_point)
    #     ignore_map = self._label_cls_data.copy()
    #     ignore_map[ignore_map == 0] = 255  # filter boundary connect label_0 and label_255
    #     ignore_map[ignore_map != 255] = 0
    #     ignore_255_boundary_map = freespace_to_fs_boundary(ignore_map, kernel_size=kernel_size+2)
    #
    #     fs_boundary_map = freespace_to_fs_boundary(freespace, kernel_size=kernel_size)
    #
    #     fs_boundary_map[ignore_255_boundary_map == 1] = 0
    #     return fs_boundary_map
    #
    def label_cls_over_img_with_dep(self, src_rate=0.4, size=None):
        img = Draw.draw_seg_over_img(img=self.img_data(size=size), seg=self.label_cls(size=size), n_classes=N_CLASSES,
                                      src_rate=src_rate)
        img_and_dep = np.concatenate([img, self._dep_data], axis=1)
        # return img
        return img_and_dep
    #
    # def pred_cls_over_img(self, src_rate=0.4, size=None, roi_box_list=None):
    #     return Draw.draw_seg_over_img(img=self.img_data(size=size), seg=self.pred_cls(size=size), n_classes=16,
    #                                   src_rate=src_rate, roi_box_list=roi_box_list)
    #
    # def fp_map_over_img(self, src_rate=0.4, size=None, roi_box_list=None):
    #     return Draw.draw_seg_over_img(img=self.img_data(size=size), seg=self.fp_map(size=size), n_classes=16,
    #                                   src_rate=src_rate, roi_box_list=roi_box_list)
    #
    # def error_map_over_img(self, src_rate=0.4, size=None, roi_box_list=None):
    #     return Draw.draw_seg_over_img(img=self.img_data(size=size), seg=self.error_map(size=size), n_classes=16,
    #                                   src_rate=src_rate, roi_box_list=roi_box_list)
    #
    # def pred_fs_boundary_over_img(self, kernel_size=3, src_rate=0.4, size=None, roi_box_list=None):
    #     fs_boundary_map = self.pred_fs_boundary_map(size=size, kernel_size=kernel_size)
    #
    #     cv2_img = self.img_data(size=size).copy()
    #     cv2_img[fs_boundary_map == 1] = cv2_img[fs_boundary_map == 1] * src_rate + np.asarray([0, 0, 255]) * (
    #             1 - src_rate)
    #
    #     if roi_box_list:
    #         for p1, p2 in roi_box_list:
    #             cv2.rectangle(cv2_img, p1, p2, color=(0, 0, 255), thickness=1)
    #     return cv2_img
