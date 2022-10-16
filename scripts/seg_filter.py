import os
import mmcv
import cv2
import numpy as np
import shutil

import torch
import torch.nn.functional as tF
import torchvision.transforms.functional as tvF

from pathlib import Path


def Hex2BGR(Hex_val):
    b = int(Hex_val[:2], 16)
    g = int(Hex_val[2:4], 16)
    r = int(Hex_val[4:6], 16)
    return [r, g, b]

color_map = {
        "ed53db": 1,  # 机动车
        "fca4f2": 2,  # 路面
        "f44e3b": 3,  # 行人
        "ffffff": 4,  # 斑马线
        # "#": 5,   # 库位线
        "adcc26": 6,  # 道路箭头
        "53d2db": 7,  # 路沿石
        "0fa835": 8,  # 墙柱
        "3fc413": 9,  # 减速带
        "7b64ff": 10,  # 车道线
        "f7112f": 11,  # 地锁(打开)
        "ed9ea6": 12,  # 轮档
        "ef8bbf": 13,  # 警示设施
        "dd8d7a": 14,  # 障碍物
        # "#": 15,  # 植草砖
        "a7bc09": 16,  # 地锁(关闭)
        "0e4168": 17,  # 非机动车
        "0f9186": 18,  # 禁停标线
        "937bce": 19,  # 自行车
        "7dc2e0": 20,  # 摩托车/自行车
        "f9d48e": 255,  # 可忽视区域

        "0": 255,  # error elem in seg mask
    }

fs_ids = [2, 4, 5, 6, 9, 10, 12, 15, 16, 18]



fs_bgrs = []

for k, v in color_map.items():
    if v in fs_ids:
        fs_bgrs.append(Hex2BGR(k))


if __name__ == '__main__':
    RootDir = Path(r'C:\Users\DELL\Desktop\seg\SegData\trainset')

    bad_dir = r'C:\Users\DELL\Desktop\seg\SegData\bad_imgs'

    print(fs_bgrs)

    # compute num_imgs
    num_mask = 0
    for casedir in RootDir.iterdir():
        for elem in os.listdir(str(casedir)):
            if elem == 'mask':
                num_mask += len(os.listdir(os.path.join(str(casedir), 'mask')))
            if elem == 'mask_gray':
                num_mask += len(os.listdir(os.path.join(str(casedir), 'mask_gray')))
    print(f"# total {num_mask} seg mask.")

    current_mask_idx = 0
    for casedir in RootDir.iterdir():
        contain_color_mask = False
        contain_gray_mask = False

        for elem in os.listdir(str(casedir)):
            if elem == 'mask':
                contain_color_mask = True
            if elem == 'mask_gray':
                contain_gray_mask = True

        save_fs_boundary_dir = os.path.join(str(casedir), 'mask_fs_boundary')

        if contain_color_mask:
            mask_dir = Path(os.path.join(str(casedir), 'mask'))
            for mask_path in mask_dir.iterdir():
                current_mask_idx += 1
                mask = cv2.imread(str(mask_path))
                h, w, c = mask.shape
                # print(mask[h//2, w//2])
                if mask[h//2, w//2].tolist() not in fs_bgrs:
                    print(mask_path)
                    mask_path = str(mask_path)
                    img_path = mask_path.replace('mask', 'image').replace('.png', '.jpg')
                    shutil.copy(mask_path, bad_dir)
                    shutil.copy(img_path, bad_dir)

        if contain_gray_mask:
            mask_dir = Path(os.path.join(str(casedir), 'mask_gray'))
            for mask_path in mask_dir.iterdir():
                mask = cv2.imread(str(mask_path))
                h, w, c = mask.shape
                # print(mask[h // 2, w // 2])
                if mask[h//2, w//2].tolist() != [0, 0, 0]:
                    print(mask_path)
                    mask_path = str(mask_path)
                    img_path = mask_path.replace('mask_gray', 'image').replace('.png', '.jpg')
                    shutil.copy(mask_path, bad_dir)
                    shutil.copy(img_path, bad_dir)