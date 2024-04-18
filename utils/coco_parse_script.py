from pycocotools.coco import COCO
import numpy as np
import cv2
import os

partition = 'train'
root_dir = '/gdata1/coco'
annFile = root_dir + '/annotations/instances_{}2014.json'.format(partition)
save_dir = root_dir + '/{}_no_crowd'.format(partition)
img_dir = root_dir + '/{}2014'.format(partition)
save_iscrowd = False

if not os.path.exists(save_dir):
    print('{} has been created!'.format(save_dir))
    os.mkdir(save_dir)

coco=COCO(annFile)
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())

nms=[cat['name'] for cat in cats]
num_cats = len(nms)
print('All {} categories.'.format(num_cats))
print(nms)

# get all images ids
imgIds = coco.getImgIds()
num_img = len(imgIds)
print('All {} images.'.format(num_img))

import time

sum_time = 0
img_cnt = 0
crowd_cnt = 0
for idx, im_id in enumerate(imgIds):  
    start_time = time.time()
    if save_iscrowd:
        annIds = coco.getAnnIds(imgIds=im_id)
    else:
        annIds = coco.getAnnIds(imgIds=im_id, iscrowd=False)
    #annIds = coco.getAnnIds(imgIds=im_id)#, iscrowd=False)        
    if len(annIds) == 0:
        continue

    image = coco.loadImgs([im_id])[0]
    # image.keys: ['coco_url', 'flickr_url', 'date_captured', 'license', 'width', 'height', 'file_name', 'id']
    h, w = image['height'], image['width']
    gt_name = image['file_name'].split('.')[0] + '.png'
    gt = np.zeros((h, w), dtype=np.uint8)

    # ann.keys: ['area', 'category_id', 'bbox', 'iscrowd', 'id', 'segmentation', 'image_id']
    anns = coco.loadAnns(annIds)
    if len(anns) == 0:
        continue
    has_crowd_flag = 0
    save_flag = 0
    for ann_idx, ann in enumerate(anns):
        cat = coco.loadCats([ann['category_id']])
        cat = cat[0]['name']
        cat = nms.index(cat) + 1    # cat_id ranges from 1 to 80

        if not ann['iscrowd']:
            #continue       
            segs = ann['segmentation']
            for seg in segs:
                seg = np.array(seg).reshape(-1, 2)    # [n_points, 2]      
                cv2.fillPoly(gt, seg.astype(np.int32)[np.newaxis, :, :], int(cat))  

        elif save_iscrowd:     
            has_crowd_flag = 1
            #continue            
            rle = ann['segmentation']['counts']
            assert sum(rle) == ann['segmentation']['size'][0] * ann['segmentation']['size'][1] 
            mask = coco.annToMask(ann)
            unique_label = list(np.unique(mask))
            assert len(unique_label) == 2 and 1 in unique_label and 0 in unique_label
            gt = gt * (1-mask) + mask * 255

    save_gt_path = os.path.join(save_dir, gt_name)
    cv2.imwrite(save_gt_path, gt)
    img_cnt += 1
    if has_crowd_flag:
        crowd_cnt += 1    

    cost_time = time.time() - start_time
    sum_time += cost_time
    avg_time = sum_time*1.0/(idx + 1)
    left_time = avg_time * (len(imgIds) - idx)/60

    if idx % 100 == 0:
        print('Processed {}/{} images. Time: {:.4f} min'.format(idx, num_img, left_time))
print('crowd/all = {}/{}'.format(crowd_cnt, img_cnt))

    



