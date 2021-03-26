# coding=utf-8

import os
import sys
sys.path.append("..")
import xml.etree.ElementTree as ET

from tqdm import tqdm

import config.yolov3_config as cfg


def parse_voc_annotation(data_path, file_type, anno_path, use_difficult_bbox = False, label_with_difficult = False):
    """
    解析Pascal VOC数据集的annotation, 每一行为"image_path (xmin,xmax,ymin,ymax,class_ind,(difficult))*N\n"
    args:
        data_path: 数据集的路径
        file_type: "trainval", "train", "val", "test", etc.
        anno_path: 标签存储路径
        use_difficult_bbox: bool = False, 是否使用difficult bbox
        label_with_difficult: bool = False, 是否登记difficult
    returns:
        len(image_ids): 数据集大小
    notes:
        xmin,xmax,ymin,ymax全部是实际值
    """
    classes = cfg.VOC_CLASS["CLASSES"]
    img_inds_file = os.path.join(data_path, "ImageSets", "Main", file_type + ".txt")
    with open(img_inds_file, "r") as f:
        lines = f.readlines()
        image_ids = [line.strip() for line in lines]

    # max_objects = 0
    # max_objects_id = ""

    with open(anno_path, "a") as f:
        for image_id in tqdm(image_ids):
            image_path = os.path.join(data_path, "JPEGImages", image_id + ".jpg")
            # 登记图片名
            annotation = image_path

            label_path = os.path.join(data_path, "Annotations", image_id + ".xml")
            root = ET.parse(label_path).getroot()
            objects = root.findall("object")

            # if len(objects) > max_objects: max_objects, max_objects_id = len(objects), image_id

            for obj in objects:
                difficult = obj.find("difficult").text.strip()
                if (not use_difficult_bbox) and (int(difficult) == 1): # difficult表示是否容易识别，0表示容易，1表示困难
                    continue
                bbox = obj.find("bndbox")
                class_ind = classes.index(obj.find("name").text.lower().strip()) # lower()用于转换为小写
                xmin = bbox.find("xmin").text.strip()
                ymin = bbox.find("ymin").text.strip()
                xmax = bbox.find("xmax").text.strip()
                ymax = bbox.find("ymax").text.strip()
                # 登记位置信息
                annotation += " " + ",".join([xmin, ymin, xmax, ymax, str(class_ind)])
                # 登记difficult
                if label_with_difficult:
                    annotation += "," + difficult
            annotation += "\n"
            # print(annotation)
            f.write(annotation)

    # VOC2007 trainval: 42, 004349
    # VOC2012 trainval: 56, 2008_007069
    # VOC2007 test: 41, 006500
    # print("max_objects = {}, max_objects_id = {}".format(max_objects, max_objects_id))

    # PASCAL VOC2007: train 2501, val 2510, test 4952
    # PASCAL VOC2012: train 5717, val 5823
    return len(image_ids)



if __name__ =="__main__":
    # trainset
    train_data_path_2007 = os.path.join(cfg.VOC_PATH, "VOC2007")
    train_data_path_2012 = os.path.join(cfg.VOC_PATH, "VOC2012")
    train_annotation_path = cfg.TRAIN_ANNO_PATH
    if os.path.exists(train_annotation_path):
        os.remove(train_annotation_path)

    # testset
    test_data_path_2007 = os.path.join(cfg.VOC_PATH, "VOC2007")
    test_annotation_path = cfg.TEST_ANNO_PATH
    if os.path.exists(test_annotation_path):
        os.remove(test_annotation_path)

    len_train = (parse_voc_annotation(train_data_path_2007, "trainval", train_annotation_path) +
                 parse_voc_annotation(train_data_path_2012, "trainval", train_annotation_path))
    len_test = parse_voc_annotation(test_data_path_2007, "test", test_annotation_path, True, True)

    print("The number of images for train and test are: train: {0} | test: {1}".format(len_train, len_test))

