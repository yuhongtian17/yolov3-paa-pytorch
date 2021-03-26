# coding=utf-8

import os
import sys
sys.path.append("..")
sys.path.append("../utils")
import random

import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import numpy as np
import cv2

import config.yolov3_config as cfg
import utils.tools as tools

from utils.augmentation import *


class LabelSmooth(object):
    def __init__(self, delta = 0.01):
        self.delta = delta

    def __call__(self, onehot, num_classes):
        return onehot * (1 - self.delta) + self.delta * 1.0 / num_classes


class UnivDataset(Dataset):
    """
    数据集类
    def __init__ args:
        mode: assert mode in ["+voc", "+coco", None], in fact choosing None would go "+coco"
        img_size: int, image size
        num_classes: int, num_classes
    """
    def __init__(self, mode, img_size, num_classes):
        super(UnivDataset, self).__init__()

        assert mode in ["+voc", "+coco", None]
        self.mode = mode
        self.img_size = img_size
        self.num_classes = num_classes

        self.__anchors = np.array(cfg.MODEL["ANCHORS"])
        self.__strides = np.array(cfg.MODEL["STRIDES"])
        self.__num_feature_map = cfg.MODEL["NUM_FEATURE_MAP"]
        self.__anchors_per_scale = cfg.MODEL["ANCHORS_PER_SCLAE"]

        if self.mode == "+voc":
            self.__annotations = self.__load_annotations("train") # 这里load进来的也全部是实际值
            self.__imgIds = None
            self.__catIds = None
            self.__annolen = len(self.__annotations)
        else:
            self.__annotations = COCO(os.path.join(cfg.COCO_PATH, "annotations", "instances_train2017.json"))
            self.__imgIds = self.__annotations.getImgIds()
            self.__catIds = sorted(self.__annotations.getCatIds())
            self.__annolen = len(self.__imgIds)

    def __len__(self):
        return self.__annolen

    def __getitem__(self, item):
        img1, bboxes1 = self.__parse_annotation(item)

        item2 = random.randint(0, self.__annolen - 1)
        item3 = random.randint(0, self.__annolen - 1)
        item4 = random.randint(0, self.__annolen - 1)

        img2, bboxes2 = self.__parse_annotation(item2)
        img3, bboxes3 = self.__parse_annotation(item3)
        img4, bboxes4 = self.__parse_annotation(item4)

        img_org, bboxes_org = Mosaic_v2(self.img_size)([img1, img2, img3, img4], [bboxes1, bboxes2, bboxes3, bboxes4])
        img_org, bboxes_org = self.__basic_data_augmentation(img_org, bboxes_org)
        img_org = img_org.transpose(2, 0, 1) # (h, w, c)维度变成(c, h, w)维度

        item_mix = random.randint(0, self.__annolen - 1)
        img_mix, bboxes_mix = self.__parse_annotation(item_mix)
        img_mix, bboxes_mix = self.__basic_data_augmentation(img_mix, bboxes_mix)
        img_mix = img_mix.transpose(2, 0, 1) # (h, w, c)维度变成(c, h, w)维度

        # 混合样本数据增强
        img, bboxes = Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
        del img_org, bboxes_org, img_mix, bboxes_mix

        # 制作标签
        index_s, index_m, index_l, label_s, label_m, label_l, bboxes_xywh = self.__creat_label(bboxes)

        img = torch.from_numpy(img).float()
        index_s = torch.from_numpy(index_s).float()
        index_m = torch.from_numpy(index_m).float()
        index_l = torch.from_numpy(index_l).float()
        label_s = torch.from_numpy(label_s).float()
        label_m = torch.from_numpy(label_m).float()
        label_l = torch.from_numpy(label_l).float()
        bboxes_xywh = torch.from_numpy(bboxes_xywh).float()

        return img, index_s, index_m, index_l, label_s, label_m, label_l, bboxes_xywh

    def __load_annotations(self, anno_type):
        """
        加载自己做好的annotation.txt文件
        notes:
            由于voc.py制作时全部是实际值，因此这里load进来的也全部是实际值
        """
        assert anno_type in ["train", "test"], "You must choice one of the \"train\" or \"test\" for anno_type parameter"
        if anno_type == "train":
            anno_path = cfg.TRAIN_ANNO_PATH
        else: # if anno_type == "test":
            anno_path = cfg.TEST_ANNO_PATH
        with open(anno_path, "r") as f:
            annotations = list(filter(lambda x:len(x) > 0, f.readlines())) # 读取行
        assert len(annotations) > 0, "No images found in {}".format(anno_path) # 要求annotation.txt文件非空

        return annotations

    def __parse_annotation(self, item):
        """
        将加载的单个annotation转换成实例
        args:
            item: int
        returns:
            img:
            bboxes:
        notes:
            bboxes依然是以图片大小为参考的实际值
        """
        if self.mode == "+voc":
            annotation = self.__annotations[item]
            anno = annotation.strip().split(" ")
            img_path = anno[0]
            img = cv2.imread(img_path)
            assert img is not None, "File Not Found " + img_path
            if len(anno) == 1: # 防止anno为空，创建一个无效框
                bboxes = np.array([[0, 0, img.shape[1], img.shape[0], -1]])
            else:
                bboxes = np.array([list(map(float, box.split(","))) for box in anno[1:]])
            # bboxes = bboxes[:, 0:5] # 有可能加入了difficult信息（但由于在voc.py中严格限制了所以不会遇到这种情况）
        else:
            img_id = int(self.__imgIds[item])
            img_path = os.path.join(cfg.COCO_PATH, "train2017", "{:012}".format(img_id) + ".jpg")
            img = cv2.imread(img_path)
            assert img is not None, "File Not Found " + img_path
            annIds = self.__annotations.getAnnIds(imgIds = [img_id], iscrowd = None)
            anns = self.__annotations.loadAnns(annIds)
            if len(anns) == 0: # 防止anns为空，创建一个无效框
                bboxes = np.array([[0, 0, img.shape[1], img.shape[0], -1]])
            else: # xmin, ymin, xmax, ymax, class_ind
                bboxes = np.array([[ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3],
                                    self.__catIds.index(ann["category_id"])] for ann in anns])

        return img, bboxes

    def __basic_data_augmentation(self, img, bboxes):
        """
        基本数据增强：随机翻转、随机裁剪、随机仿射
        """
        img, bboxes = RandomHorizontalFilp()(np.copy(img), np.copy(bboxes))
        img, bboxes = RandomCrop()(np.copy(img), np.copy(bboxes))
        img, bboxes = RandomAffine_v2()(np.copy(img), np.copy(bboxes))
        img, bboxes = Resize((self.img_size, self.img_size), correct_box = True)(np.copy(img), np.copy(bboxes))

        return img, bboxes

    def __creat_label(self, bboxes):
        """
        """
        # 将bbox中超出原图的部分裁掉
        # 
        # 在COCO数据集中，有很多与边界重合的框的xmax, ymax数值标记为框的大小数值。
        # 这会使得在计算xind, yind时，可能出现“恰好越界”的错误。
        # 因此在创建label阶段将其修正。
        # 
        # 类似修正亦可见于eval/evaluator.py。
        # 
        bboxes = np.concatenate([np.maximum(bboxes[:, 0:2], [0, 0]),
                                 np.minimum(bboxes[:, 2:4], [self.img_size - 1, self.img_size - 1]),
                                 bboxes[:, 4:]], axis = -1)

        train_output_size = self.img_size / self.__strides

        index = []
        label = []

        max_objects = 100 # 从voc.py和coco.py的结果我们知道一张图不会超过100个框
        bboxes_xywh = np.zeros((max_objects, 6 + self.num_classes))
        bbox_count = 0

        for bbox in bboxes:
            # (0) 去除无效的bbox
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3] or bbox[4] < 0 or bbox[4] >= self.num_classes:
                continue

            # (1) 创建bbox的标签
            # bbox: [xmin, ymin, xmax, ymax, class_ind, confidence]. 均是实际值
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])
            bbox_mix = bbox[5]
            # bbox_xywh: [x, y, w, h]. 均是实际值
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis = -1)
            # bbox_xywh_scaled: [[x, y, w, h] / 8, [x, y, w, h] / 16, [x, y, w, h] / 32]. 不同尺寸的特征图上, 仍均是实际值
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.__strides[:, np.newaxis]
            # 将类别转换为one-hot编码
            one_hot = np.zeros(self.num_classes, dtype = np.float32)
            one_hot[bbox_class_ind] = 1.0
            one_hot_smooth = LabelSmooth()(one_hot, self.num_classes) # 标签平滑化（而不是极端的0和1）

            # (2) 写入bboxes_xywh
            bboxes_xywh[bbox_count % max_objects, 0:4] = bbox_xywh
            bboxes_xywh[bbox_count % max_objects, 4:5] = 1.0
            bboxes_xywh[bbox_count % max_objects, 5:6] = bbox_mix
            bboxes_xywh[bbox_count % max_objects, 6:] = one_hot_smooth
            bbox_count += 1

        for fmap in range(self.__num_feature_map): # 对于每种尺寸
            # 创建锚框集
            anchors_fmap = self.__create_anchors_famp(train_output_size[fmap], self.__anchors_per_scale, self.__anchors[fmap])
            anchors_fmap = anchors_fmap[:, :, :, np.newaxis, :] # 四维拓展成五维以满足广播机制
            # 创建ground truth集
            gt_fmap = bboxes_xywh[:, 0:4] / self.__strides[fmap]
            # 计算IoU
            iou = tools.iou_xywh_numpy(anchors_fmap, gt_fmap)
            # 取IoU最大的那个，返回索引
            iou_max = iou.argmax(-1)
            index.append(iou_max)
            # 根据索引创建label
            label_fmap = bboxes_xywh[iou_max]
            label.append(label_fmap)

        index_s, index_m, index_l = index
        label_s, label_m, label_l = label

        return index_s, index_m, index_l, label_s, label_m, label_l, bboxes_xywh

    def __create_anchors_famp(self, num_grids, num_anchor, anchors):
        anchors_list = []
        for i in range(num_anchor):
            y = np.arange(0.5, num_grids + 0.5)
            y = y[:, np.newaxis, np.newaxis]
            y = y.repeat(num_grids, 1)
            # print(y)
            x = np.arange(0.5, num_grids + 0.5)
            x = x[np.newaxis, :, np.newaxis]
            x = x.repeat(num_grids, 0)
            # print(x)
            wh = np.copy(anchors[i])
            wh = wh[np.newaxis, np.newaxis, :]
            wh = wh.repeat(num_grids, 0)
            wh = wh.repeat(num_grids, 1)
            # print(wh)
            grid_xywh = np.concatenate([x, y, wh], -1)
            # print(grid_xywh)
            anchors_list.append(grid_xywh)
        anchors_fmap = np.stack(anchors_list, -2) # shape: [num_grids, num_grids, num_anchor, xywh]
        # print(anchors_fmap)
        return anchors_fmap

