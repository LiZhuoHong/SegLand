import os
from shutil import copyfile
import rasterio
import numpy as np
import copy

if __name__ == '__main__':
    class_names = ['Vehicle & cargo-trailer', 'Parking space', 'Sports field', 'Building type 2']
    indexes = [8, 9, 10, 11]  
    all_list = []
    for i, class_name in enumerate(class_names):
        index = indexes[i]
        print(index, class_name)
        ori_path = 'YOUR_PATH_OF_CUTMIX_SAMPLES' + class_name
        dis_path = 'YOUR_PATH_OF_CUTMIX_SAMPLES'
        ori_path_image = os.path.join(ori_path, class_name + '_image')
        ori_path_label = os.path.join(ori_path, class_name + '_mask')
        dis_path_image = os.path.join(dis_path, 'image')
        dis_path_label = os.path.join(dis_path, 'label')
        if not os.path.exists(dis_path_image):
            os.mkdir(dis_path_image)
        if not os.path.exists(dis_path_label):
            os.mkdir(dis_path_label)

        for root, dirs, files in os.walk(ori_path_label):
            for file in files:
                out_fn = file.split('.')[0][:-2]
                ori_fn = os.path.join(root, file)
                out_fn_1 = copy.deepcopy(out_fn)
                out_fn_1 = out_fn_1 + "_new"
                while out_fn_1 in all_list:
                    out_fn_1 = out_fn_1 + 'a'
                dis_fn = os.path.join(dis_path_label, out_fn_1 + '.tif')
                f = rasterio.open(ori_fn)
                data = f.read()
                output_data = np.where(data == 30, index, data)
                output_profile = f.profile.copy()
                output_profile["driver"] = "GTiff"
                output_profile["dtype"] = "uint8"
                with rasterio.open(dis_fn, "w", **output_profile) as g:
                    g.write(output_data[0], 1)
                ori_fn_image = os.path.join(ori_path_image, out_fn + '.tif')
                if os.path.exists(ori_fn_image):
                    dis_fn_image = os.path.join(dis_path_image, out_fn_1 + '.tif')
                    copyfile(ori_fn_image, dis_fn_image)
                all_list.append(out_fn_1)

    fns = open(os.path.join(dis_path, 'train.txt'), 'w')
    all_list = [fn + '\n' for fn in all_list]
    fns.writelines(all_list)

    for fn in all_list:
        fn1 = fn.strip() + '.tif'
        if not os.path.exists(os.path.join(dis_path, 'image', fn1)):
            print(fn1)
            print("not exist!")
        if not os.path.exists(os.path.join(dis_path, 'label', fn1)):
            print("not exist!")