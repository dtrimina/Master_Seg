# TODO 代码优化
import torch
import torchvision.transforms.functional as functional
import math
import numpy as np
import cv2

FONT = cv2.FONT_HERSHEY_TRIPLEX
FONT_SIZE = 0.5
FONT_WEIGHT = 1
LINE_SIZE = 1


def color_map(N=256, normalized=False):
    """
    Return Color Map in PASCAL VOC format
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255.0 if normalized else cmap
    return cmap


def class_to_RGB(label, N, cmap=None, normalized=False):
    '''
        label: 2D numpy array with pixel-level classes shape=(h, w)
        N: number of classes, including background, should in [0, 255]
        cmap: list of colors for N class (include background) \
              if None, use VOC default color map.
        normalized: RGB in [0, 1] if True else [0, 255] if False

        :return 上色好的3D RGB numpy array shape=(h, w, 3)
    '''
    dtype = "float32" if normalized else "uint8"

    assert len(label.shape) == 2, f'label should be 2D, not {len(label.shape)}D'
    label_class = np.asarray(label)

    label_color = np.zeros((label.shape[0], label.shape[1], 3), dtype=dtype)

    if cmap is None:
        # 0表示背景为[0 0 0]黑色,1~N表示N个类别彩色
        cmap = color_map(N, normalized=normalized)
    else:
        cmap = np.asarray(cmap, dtype=dtype)
        cmap = cmap / 255.0 if normalized else cmap

    assert cmap.shape[0] == N, f'{N} classes and {cmap.shape[0]} colors not match.'

    # 给每个类别根据color_map上色
    for i_class in range(N):
        label_color[label_class == i_class] = cmap[i_class]

    return label_color


def draw_seg_over_img(img, seg, color_map, n_classes, ignore_index=255, source_image_ration=0.4):
    img = img.copy()
    # label_img = seg
    row, col = seg.shape[0:2]

    colors = np.zeros((row, col, 3), dtype=np.uint8)

    for i in range(n_classes):
        colors[seg == i] = color_map[i]
    colors[seg == ignore_index] = img[seg == ignore_index]
    img = img * source_image_ration + colors * (1 - source_image_ration)
    img = img.astype(np.uint8)

    return img


def draw_fs_boundary_over_img(img, seg, color_map=None, source_image_ration=0.4):
    img = img.copy()

    img[seg == 1] = img[seg == 1] * source_image_ration + np.asarray([0, 0, 255]) * (1 - source_image_ration)

    return img
