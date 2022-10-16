import os
import csv

src_txt_file = r'E:\work\Master_Seg\Space_seg\result\Seg_model_baseline_v3fullconv_2022-07-13-22-04-576x576_12m\result_TestSet12M_4m_area.txt'

with open(src_txt_file, 'r', encoding='utf-8') as fp:
    infos = fp.readlines()

with open('result.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    for i, info in enumerate(infos):
        print(i, info)
        if info.startswith('test') or info.startswith('case'):
            info = [info.strip()]
        else:
            info = [i.strip() for i in info.split(',')]

        writer.writerow(info)


