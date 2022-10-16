from data import SegImage
import os
import cv2

case_dir = r'E:\Dataset\SegData\Trainset_src\[case29]-[n1199]-[12m]'

img_dir = os.path.join(case_dir, 'image')
mask_dir = os.path.join(case_dir, 'mask_gray')

name_list = sorted(os.listdir(img_dir))

index = 0
while index >= 0:
    name = name_list[index]
    print(f'[{index}/{len(name_list)}]: {name}')


    img_path = os.path.join(img_dir, name)
    mask_path = os.path.join(mask_dir, name.replace('.jpg', '.png'))

    segimg = SegImage(img_path, mask_path)

    img_show = segimg.label_cls_over_img(src_rate=0.4)
    img_show = cv2.resize(img_show, dsize=(960, 960))

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