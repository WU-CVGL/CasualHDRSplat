import os
import shutil

if __name__ == '__main__':
    root = '/home/cvgluser/blender/blender-3.6.13-linux-x64/data/deblurnerf/rawdata_new_tra1/cozyroom'
    blur_path = os.path.join(root, 'blurtrajectories')
    image_test_path = os.path.join(root, 'images_test')
    os.makedirs(image_test_path, exist_ok=True)
    indexs = os.listdir(blur_path)
    for index in indexs:
        src_img_path = os.path.join(blur_path, index, 'ldr', '000000.png')
        dst_img_path = os.path.join(image_test_path, f"frame_{int(index):05d}.png")
        shutil.copy(src_img_path, dst_img_path)
