import os
import mmcv
import cv2
import numpy as np

case_dir = r'E:\Dataset\SegData\Trainset_src\[case20]-[n466]-[12m]-[changcheng]'

img_dir = os.path.join(case_dir, 'image')
mask_dir = os.path.join(case_dir, 'mask')

name_list = sorted(os.listdir(img_dir))

index = 0
while index >= 0:
    name = name_list[index]

    img_path = os.path.join(img_dir, name)
    mask_path = os.path.join(mask_dir, name.replace('.jpg', '.png'))

    image = mmcv.imread(img_path)
    mask = mmcv.imread(mask_path)
    img_show = (image * 0.4 + mask * 0.6).astype(np.uint8)
    img_show = mmcv.imresize(img_show, size=(768, 768))

    cv2.imshow('img', img_show)

    key = cv2.waitKeyEx()

    if index == 0:
        cv2.waitKey()

    if key == 13:  # press enter
        pass

    elif key == 97:  # press left
        if index >= 1:
            index -= 1

    elif key == 100:  # press any other keys
        if index < len(name_list) - 1:
            index += 1

    elif key == 32:
        cv2.waitKey()

    else:
        if index < len(name_list) - 1:
            index += 1