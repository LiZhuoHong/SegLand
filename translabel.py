import rasterio
import os

colormap = {
    0:  (147, 147, 147),
    1:  (49, 139, 87),
    2: (0, 255, 0),
    3: (128, 0, 0),
    4: (75, 181, 73),
    5: (245, 245, 245),
    6: (35, 91, 200),
    7: (247, 142, 82),
    8:  (166, 166, 171),
    9:  (3, 7, 255),
    10: (255, 242, 0),
    11: (170, 255, 0),
}

if __name__ == '__main__':
    label_dir = '/home/ashelee/Original_OEM/ori_set/label'
    save_dir = '/home/ashelee/Original_OEM/ori_set/label_color'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file_names = []
    for root, dirs, files in os.walk(label_dir):
        for file in files:
            file_names.append(file)
    for f in file_names:
        fn = os.path.join(label_dir, f)
        h = rasterio.open(fn)
        label = h.read()
        output_profile = h.profile.copy()
        output_profile["driver"] = "GTiff"
        output_profile["dtype"] = "uint8"
        output_profile["count"] = 1
        output_profile["nodata"] = 0
        output_fn = os.path.join(save_dir, f)
        with rasterio.open(output_fn, "w", **output_profile) as g:
            g.write(label[0], 1)
            g.write_colormap(1, colormap)