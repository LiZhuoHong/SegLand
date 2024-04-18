import os
import scipy
import numpy as np
from PIL import Image

colormap = {
    0:  (147, 147, 147),
    1:  (49, 139, 87),
    2: (0, 255, 0),
    3: (128, 0, 0),
    4: (75, 181, 73),
    5: (245, 245, 245),
    6: (35, 91, 200),
    7: (247, 142, 82)
}

if __name__ == '__main__':
    colormap = np.array([[147, 147, 147], 
                         [ 49, 139,  87],
                         [  0, 255,   0], 
                         [128,   0,   0],
                         [ 75, 181,  73], 
                         [245, 245, 245],
                         [ 35,  91, 200], 
                         [247, 142,  82]]).astype(np.uint8)
    fusion_list = [
        'PATH_OF_PROBABILITY_MAPS_FOR_FUSION_1',
        'PATH_OF_PROBABILITY_MAPS_FOR_FUSION_2',
        'PATH_OF_PROBABILITY_MAPS_FOR_FUSION_3',
        '...'
    ]
    output_path = 'PATH_OF_OUTPUT_PROBABILITY_MAPS'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    fns = []
    mats = []
    for fusion_path in fusion_list:
        for root, dirs, files in os.walk(fusion_path):
            for file in files:
                fn = os.path.join(root, file)
                prob = scipy.io.loadmat(fn)['outputs'][0]
                if file not in fns:
                    fns.append(file)
                    mats.append(prob)
                else:
                    idx = fns.index(file)
                    mats[idx] += prob
    mats = [np.argmax(mat / len(fusion_list), axis=0) for mat in mats]
    for i in range(len(fns)):
        out_image = Image.fromarray(mats[i].astype(np.uint8), 'P')
        out_image = out_image.resize((1024, 1024), Image.NEAREST)
        out_image.putpalette(colormap)
        out_image.save(os.path.join(output_path, fns[i].split('.')[0] + '.png'))