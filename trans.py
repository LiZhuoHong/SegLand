from tqdm import tqdm
import os
import numpy as np
from PIL import Image

# 定义文件夹路径
folder_path = "/home/lufangxiao/POP-main/output"
save_path ="/home/lufangxiao/POP-main/upload"
if not os.path.exists(save_path):
    os.mkdir(save_path)
# 获取文件夹中所有png文件
png_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]

# 上采样
for file_name in tqdm(png_files):
    img_path = os.path.join(folder_path, file_name)
    img = Image.open(img_path)
    img = img.resize((1024, 1024), Image.NEAREST)
    save_path_1 = os.path.join(save_path, file_name)
    save_path_1 = os.path.splitext(save_path_1)[0] + ".png"
    # print(save_path)
    img.save(save_path_1)

print("处理完成！")