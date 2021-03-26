# coding=utf-8

import os
import shutil
import json
import tempfile

import torch
# import torchvision.transforms as transforms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import cv2
from tqdm import tqdm

import config.yolov3_config as cfg
from utils.datasets import *
from utils.gpu import *
from utils.tools import *
from utils.visualize import *


def calc_ap(annotations, pred_path, class_ind, iou_thresh = 0.5, use_07_metric = False):
    """
    计算AP
    args:
        annotations: 读取好的gt数据。gt的每一行为"img_path (xmin,xmax,ymin,ymax,class_ind,difficult)*N\n"。全部一起存储
        pred_path: pred数据存储路径。pred的每一行为"图片名（不含扩展名） 评分 xmin xmax ymin ymax\n"。按类别存储
        class_ind: 要计算AP的类别index
        iou_thresh: float = 0.5。与gt的iou不低于阈值的pred才会被认为是tp
        use_07_metric: bool = False。使用十一段积分而不是直接积分求AP
    returns:
        rec:
        prec:
        ap:
    notes:
        original information:
        # --------------------------------------------------------
        # Fast/er R-CNN
        # Licensed under The MIT License [see LICENSE for details]
        # Written by Bharath Hariharan
        # --------------------------------------------------------
    """
    # 读取gt框
    class_recs = {}
    npos = 0

    for annotation in annotations:
        anno = annotation.strip().split(" ")                            # 按空格分割出gt框
        img_path = anno[0]
        bboxes = anno[1:]
        # bboxes = np.array([bbox.strip().split(",") for bbox in bboxes]).astype(np.int32)
        bboxes = [bbox.strip().split(",") for bbox in bboxes]           # 按","分割出具体信息
        bboxes = np.array([[int(x) for x in bbox] for bbox in bboxes])  # 转换成np.array(dtype=int)
        class_mask = bboxes[:, 4] == class_ind
        bboxes = bboxes[class_mask]
        bbox = bboxes[:, 0:4].astype(np.float32)
        difficult = bboxes[:, 5].astype(np.bool)
        det = [False] * len(bboxes)
        npos = npos + sum(~difficult)
        class_recs[img_path] = {
            "bbox": bbox,
            "difficult": difficult,
            "det": det}

    # 读取pred框
    with open(pred_path, "r") as f:                                     # 打开文件，按行读取
        lines = f.readlines()
    assert len(lines) > 0, "No images found in {}".format(pred_path)    # 要求pred_<class_name>.txt文件非空

    splitlines = [x.strip().split(" ") for x in lines]                  # 按空格划分
    image_ids = [x[0] for x in splitlines]                              # 图片名
    confidence = np.array([float(x[1]) for x in splitlines])            # 评分
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])      # 框的位置信息

    sorted_ind = np.argsort(-confidence)                                # 按评分降序排列
    # sorted_scores = np.sort(-confidence)
    image_ids = [image_ids[x] for x in sorted_ind]                      # 根据降序index重排图片名
    BB = BB[sorted_ind, :]                                              # 根据降序index重排框的位置信息

    nd = len(image_ids)                                                 # nd：预测框的总数
    tp = np.zeros(nd)                                                   # tp登记簿
    fp = np.zeros(nd)                                                   # fp登记簿
    for d in range(nd):
        R = class_recs[image_ids[d]]                                    # 对于一个框的图片名，到class_recs中去查找对应项
        bb = BB[d, :]#.astype(float)                                    # pred的位置信息
        iou_max = -np.inf
        BBGT = R["bbox"]#.astype(float)                                 # gt的位置信息

        if BBGT.size > 0:                                               # 对所有的gt同时计算iou找出最大的
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            intersection = iw * ih

            # union
            union = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - intersection)

            iou = intersection / union
            iou_max = np.max(iou)
            jmax = np.argmax(iou)

        if iou_max > iou_thresh:                                        # 如果超过了iou阈值
            if not R['difficult'][jmax]:                                # （如果difficult == False）
                if not R['det'][jmax]:                                  # 如果该gt仍未对应pred
                    tp[d] = 1.                                          # 记为tp
                    R['det'][jmax] = True                               # 该gt对应pred
                else:
                    fp[d] = 1.                                          # 否则记为fp
        else:
            fp[d] = 1.                                                  # 否则记为fp

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float32).eps)
    
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return rec, prec, ap


class Evaluator(object):
    def __init__(self, model, classes, num_classes, visual = True):
        self.model = model
        self.classes = classes
        self.num_classes = num_classes
        self.visual = visual

        self.__pred_result_path = os.path.join(cfg.PROJECT_PATH, "data", "temp")
        # 创建新的temp目录
        if os.path.exists(self.__pred_result_path):
            shutil.rmtree(self.__pred_result_path)
        os.mkdir(self.__pred_result_path)

        self.__test_img_size = cfg.TEST["TEST_IMG_SIZE"]
        self.__conf_thresh = cfg.TEST["CONF_THRESH"]
        self.__iou_thresh = cfg.TEST["IOU_THRESH"]

        self.__device = next(model.parameters()).device


    def APs_voc(self, use_tqdm = True, multi_scale_test = False, flip_test = False, AP_thresh = 0.5, use_07_metric = False):
        """
        对于特定的测试集及对应标注数据计算AP（这里指VOCtest_06-Nov-2007）
        args:
            use_tqdm: bool = True
            multi_scale_test: bool = False
            flip_test: bool = False
            AP_thresh: float = 0.5
            use_07_metric: bool = False
        """
        annotations = self.__load_annotations("test")

        for class_name in self.classes:
            f = open(os.path.join(self.__pred_result_path, "pred_" + class_name + ".txt"), "a") # 防止网络性能过差时pred文件不存在

        visual_count = 0

        # 进度条方式显示
        for annotation in (tqdm(annotations) if use_tqdm else annotations):
            # 读取图片
            anno = annotation.strip().split(" ")
            img_path = anno[0]
            img = cv2.imread(img_path)
            # 获取bbox
            bboxes_prd = self.get_bbox(img, multi_scale_test, flip_test)

            # 用100张图测试能否正常输出
            if bboxes_prd.shape[0] != 0 and self.visual and visual_count < 100:
                # 以下这段代码亦可见于test.py
                boxes = bboxes_prd[..., :4]
                class_inds = bboxes_prd[..., 5].astype(np.int32)
                scores = bboxes_prd[..., 4]

                visualize_boxes(image = img, boxes = boxes, labels = class_inds, probs = scores, class_labels = self.classes)
                path = os.path.join(self.__pred_result_path, "{}.jpg".format(visual_count))
                cv2.imwrite(path, img)

                visual_count += 1

            # 将预测结果按类别追加写入self.__pred_result_path下的txt文件里去
            # 格式是"图片名（不含扩展名） 评分 xmin xmax ymin ymax\n"
            for bbox in bboxes_prd:
                coor = np.array(bbox[:4], dtype = np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])

                class_name = self.classes[class_ind]
                score = "%.4f" % score
                xmin, ymin, xmax, ymax = map(str, coor)
                s = " ".join([img_path, score, xmin, ymin, xmax, ymax]) + "\n"

                with open(os.path.join(self.__pred_result_path, "pred_" + class_name + ".txt"), "a") as f:
                    f.write(s)

        # 计算AP
        APs = {}

        for class_ind in range(self.num_classes):
            class_name = self.classes[class_ind]
            pred_path = os.path.join(self.__pred_result_path, "pred_" + class_name + ".txt")
            R, P, AP = calc_ap(annotations, pred_path, class_ind, AP_thresh, use_07_metric)
            APs[class_name] = AP

        return APs


    def APs_coco(self, use_tqdm = True):
        """
        """
        coco_gt = COCO(os.path.join(cfg.COCO_PATH, "annotations", "instances_val2017.json"))
        coco_dt = COCO(os.path.join(cfg.COCO_PATH, "annotations", "instances_val2017.json"))

        imgIds = coco_gt.getImgIds() # 图片标识符的顺序总list
        catIds = sorted(coco_gt.getCatIds()) # 类别标识符的顺序总list
        bboxes_prd_coco = []

        visual_count = 0

        # 进度条方式显示
        for imgId in (tqdm(imgIds) if use_tqdm else imgIds):
            # （按图片标识符顺序）读取图片
            img_id = int(imgId)

            img_path = os.path.join(cfg.COCO_PATH, "val2017", "{:012}".format(img_id) + ".jpg")
            img = cv2.imread(img_path)
            # 获取bbox
            bboxes_prd = self.get_bbox(img)

            # 用100张图测试能否正常输出
            if bboxes_prd.shape[0] != 0 and self.visual and visual_count < 100:
                # 以下这段代码亦可见于test.py
                boxes = bboxes_prd[..., :4]
                class_inds = bboxes_prd[..., 5].astype(np.int32)
                scores = bboxes_prd[..., 4]

                visualize_boxes(image = img, boxes = boxes, labels = class_inds, probs = scores, class_labels = self.classes)
                path = os.path.join(self.__pred_result_path, "{}.jpg".format(visual_count))
                cv2.imwrite(path, img)

                visual_count += 1

            # 将预测结果追加写入bboxes_prd_coco
            for bbox in bboxes_prd:
                catId = catIds[int(bbox[5])]
                bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                score = bbox[4]
                A = {
                    "image_id": imgId, # 图片标识符
                    "category_id": catId, # 类别标识符
                    "bbox": bbox_xywh,
                    "score": score,
                    "segmentation": []}
                bboxes_prd_coco.append(A)

        # 计算AP
        annType = ["segm", "bbox", "keypoints"]

        if len(bboxes_prd_coco) > 0:
            _, tmp = tempfile.mkstemp()
            json.dump(bboxes_prd_coco, open(tmp, "w"))
            coco_dt = coco_dt.loadRes(tmp)
            coco_eval = COCOeval(coco_gt, coco_dt, annType[1])
            coco_eval.params.imgIds = imgIds
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            return coco_eval.stats
        else:
            return [0.0, 0.0, 0.0]


    def get_bbox(self, img, multi_scale_test = False, flip_test = False):
        """
        输入图片和测试尺度要求，输出预测框（经过NMS的）
        returns:
            bboxes: [:, (xmin, ymin, xmax, ymax, score, class)]的格式
        """
        # 如果是多尺度测试
        if multi_scale_test:
            test_input_sizes = range(320, 640, 96)
            bboxes_list = []
            for test_input_size in test_input_sizes:
                bboxes_list.append(self.__predict(img, test_input_size))
                if flip_test:
                    bboxes_flip = self.__predict(img[:, ::-1], test_input_size)
                    bboxes_flip[:, [0, 2]] = img.shape[1] - bboxes_flip[:, [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes = np.row_stack(bboxes_list)
        # 否则就仅将单一尺度的图片作为输入进行测试
        else:
            bboxes = self.__predict(img, self.__test_img_size)

        # NMS
        bboxes = nms(bboxes, self.__conf_thresh, self.__iou_thresh)
        # 附加要求，最多100个框
        if len(bboxes) > 100:
            confidence = bboxes[:, 4]
            sorted_ind = np.argsort(-confidence)
            bboxes = bboxes[sorted_ind[:100], :]

        return bboxes


    def __predict(self, img, test_img_size):
        """
        输入图片和变换尺寸，输出预测框（未经NMS的，这是因为多尺度测试需要所有框一起NMS）
        """
        org_img = np.copy(img) # 创建副本
        org_h, org_w, _ = org_img.shape # 获取高与宽

        img = Resize((test_img_size, test_img_size))(img).transpose(2, 0, 1) # (h, w, c)维度变成(c, h, w)维度
        img = torch.from_numpy(img[np.newaxis, ...]).float() # 转换为tensor # 网络权重是float类型，所以图像转tensor也要float类型（？）
        # 用transforms.ToTensor()速度会变慢
        # img = Resize((test_img_size, test_img_size))(img)
        # img = transforms.ToTensor()(img).unsqueeze(0).float()
        img = img.to(self.__device)

        self.model.eval() # 加载一个evaluation模式的model
        with torch.no_grad():
            _, p_d = self.model(img) # 数据组织形式详见yolov3.py、yolo_head.py（p_d输出是以原图大小为标准的值）
        pred_bbox = p_d.squeeze().cpu().numpy() # pred_bbox是[:, (x, y, w, h, confi) + 20]的格式
        bboxes = self.__convert_pred(pred_bbox, org_h, org_w, test_img_size) # 把pred_bbox转换为bboxes的形式即[:, (xmin, ymin, xmax, ymax, score, class)]的格式

        return bboxes


    def __convert_pred(self, pred_bbox, org_h, org_w, test_img_size):
        """
        转换与过滤预测框
        """
        pred_coor = xywh2xyxy(pred_bbox[:, :4])
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # (1) 将bbox转换回原图上
        # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
        # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
        # 需要注意的是，无论我们在训练的时候使用什么数据增强方式，都不影响此处的转换方式
        # 假设我们对输入测试图片使用了转换方式A，那么此处对bbox的转换方式就是方式A的逆向过程
        resize_ratio = min(1.0 * test_img_size / org_w, 1.0 * test_img_size / org_h)
        dw = (test_img_size - resize_ratio * org_w) / 2
        dh = (test_img_size - resize_ratio * org_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # (2) 将预测的bbox中超出原图的部分裁掉
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]), np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis = -1)

        # (3) 保留范围有效的bbox
        valid_mask = np.logical_and((pred_coor[:, 0] < pred_coor[:, 2]), (pred_coor[:, 1] < pred_coor[:, 3]))

        # (4) 保留score不低于threshold的bbox
        pred_class = np.argmax(pred_prob, axis = -1)
        pred_score = pred_conf * pred_prob[np.arange(len(pred_coor)), pred_class] # np.arange()用于创建等差数组
        score_mask = pred_score >= self.__conf_thresh

        mask = np.logical_and(valid_mask, score_mask)

        coors = pred_coor[mask]
        scores = pred_score[mask]
        classes = pred_class[mask]

        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis = -1) # np.concatenate()在-1维上进行拼接

        return bboxes


    # 与utils/datasets.py相同
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

