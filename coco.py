import os

from pycocotools.coco import COCO
import cv2
from tqdm import tqdm

import config.yolov3_config as cfg


annotations = COCO(os.path.join(cfg.COCO_PATH, "annotations", "instances_train2017.json"))
#coco = annotations

imgIds = annotations.getImgIds()
#anns_len_max = 0
#anns_zero_count = 0

#aspect_ratio_max = 0
#hw = (0, 0)

for imgId in tqdm(imgIds):
    img_id = int(imgId)
    annIds = annotations.getAnnIds(imgIds = [img_id], iscrowd = None)
    anns = annotations.loadAnns(annIds)

    #anns_len = len(anns)
    #if anns_len > anns_len_max:
    #    anns_len_max = anns_len
    #if anns_len == 0:
    #    anns_zero_count += 1

    img_path = os.path.join(cfg.COCO_PATH, "train2017", "{:012}".format(img_id) + ".jpg")
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    #aspect_ratio = max(h / w, w / h)
    #if aspect_ratio > aspect_ratio_max:
    #    aspect_ratio_max = aspect_ratio
    #hw = (h, w)

    for ann in anns:
        if ann["bbox"][0] < 0:
            print(img_id, "0")
        if ann["bbox"][1] < 0:
            print(img_id, "1")
        if ann["bbox"][2] > w:
            print(img_id, "2")
        if ann["bbox"][3] > h:
            print(img_id, "3")


# 两者是一样的
# print(id(annotations), id(coco))
# train2017: 93 1021 / 118287
# val2017: 63 48 / 5000
# print(anns_len_max, anns_zero_count)
# (375, 500)
# print(aspect_ratio_max, hw)

