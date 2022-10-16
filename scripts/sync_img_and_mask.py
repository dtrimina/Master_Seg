import os
import shutil

img_dir = r'C:\Users\DELL\Desktop\6.30-4973#完成\image'
mask_dir = r'C:\Users\DELL\Desktop\6.30-4973#完成\seg\mask'

tar_dir = r'C:\Users\DELL\Desktop\SegData'

for mask_name in os.listdir(mask_dir):

    mask_path = os.path.join(mask_dir, mask_name)
    img_path = os.path.join(img_dir, mask_name.replace('.png', '.jpg'))

    shutil.copy(img_path, tar_dir)
    shutil.copy(mask_path, tar_dir)



