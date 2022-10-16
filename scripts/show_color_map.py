# -*- encoding: utf-8 -*-

from common import CLASSES

import cv2
import numpy as np

spaceos_color = [
		[ 0, 0, 0 ],		#label 0
		[ 255, 255, 0 ],	#label 1 car
		[ 0, 255, 0 ],		#label 2 road
		[ 0, 255, 255 ],	#label 3 ped
		[ 128, 128, 255 ],	#label 4 zebra
		[ 0, 64, 128 ],		#label 5 parkline
		[ 64, 128, 128 ],	#label 6 roadarrow
		[ 0, 0, 255 ],		#label 7 curbstone
		[ 64, 64, 0 ],		#label 8 wallcolumn
		[ 0, 100, 0 ],		#label 9 speed bummp
		[ 255, 0, 128 ],	#label 10 lane line
		[ 128, 128, 128 ],	#label 11 park lock
		[ 0, 0, 255 ],		#label 12 vehicle stoper
		[ 255, 128, 128 ],	#label 13 traffic cone
		[ 64, 255, 255 ],	#label 14 obstacle
		[ 64, 255, 128 ],	#label 15 park area
	]

PALETTE = spaceos_color

if __name__ == '__main__':

    one_line_shape = (50, 400)

    color_map = np.zeros((one_line_shape[0] * len(PALETTE), one_line_shape[1], 3), dtype=np.uint8)

    for i, (class_name, color) in enumerate(zip(CLASSES, PALETTE)):
        start_row = i * 50
        color_map[i*50: (i+1)*50, 100:] = color
        cv2.putText(color_map, class_name, (5, i*50 + 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.6,  # 字体大小
                    (255, 0, 255),  # 颜色
                    1,  # 用于绘制文本的线条的厚度厚度。
                    cv2.LINE_AA)  # 线型形状

    cv2.imshow('color_map', color_map)
    cv2.waitKey()
