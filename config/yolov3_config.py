# coding=utf-8

import os

DATASET_PATH = "/home/ubuntu/Workspace/YuHongtian/Dataset"
VOC_PATH = os.path.join(DATASET_PATH, "VOCdevkit0712")
COCO_PATH = os.path.join(DATASET_PATH, "MSCOCO2017")

PROJECT_PATH = "/home/ubuntu/Workspace/YuHongtian/YOLOv3"
TRAIN_ANNO_PATH = os.path.join(PROJECT_PATH, "data", "train_annotation.txt")
TEST_ANNO_PATH = os.path.join(PROJECT_PATH, "data", "test_annotation.txt")

# class
VOC_CLASS = {
    "CLASSES": (
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor"),
    "NUM_CLASSES": 
        20}

COCO_CLASS = {
    "CLASSES": (
        "person",
        "bicycle",
        "car",
        "motorbike",
        "aeroplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "sofa",
        "pottedplant",
        "bed",
        "diningtable",
        "toilet",
        "tvmonitor",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
        "__background__"),
    "NUM_CLASSES": 
        80}

# model
MODEL = {
    "ANCHORS": (
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
        # anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
        #                /8      /8      /8      /16     /16      /16      /32       /32       /32
        ((1.25, 1.625), (2.0, 3.75), (4.125, 2.875)),               # Anchors for small obj
        ((1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)),       # Anchors for medium obj
        ((3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875))),   # Anchors for big obj
    "STRIDES": 
        (8, 16, 32),
    "NUM_FEATURE_MAP":
        3,
    "ANCHORS_PER_SCLAE": 
        3}

# train
TRAIN = {
    "TRAIN_IMG_SIZE":       608, # 多尺度训练的最大尺度，防止内存不足
    "BATCH_SIZE":           8,
    "NUM_WORKERS":          0,
    "EPOCHS":               50,
    "LR_DECAY_EPOCHS":      (30, 40),
    "WARMUP_EPOCHS":        2,
    "MOMENTUM":             0.9,
    "WEIGHT_DECAY":         0.0005,
    "LR_INIT":              1e-4,
    "LR_END":               1e-6,
    "IOU_THRESHOLD_LOSS":   0.5, # 用于计算loss时的noobj判断标准（iou_max < iou_thresh）
    "MULTI_SCALE_TRAIN":    True,
    "AUGMENT":              True}

# test
TEST = {
    "TEST_IMG_SIZE":        416, # 32 * 13
    "BATCH_SIZE":           1,
    "NUM_WORKERS":          0,
    "CONF_THRESH":          0.05,
    "IOU_THRESH":           0.5, # 用于nms时的重叠框去除标准
    # 另：
    # 计算AP时的iou_thresh是独立指定的（见eval.evaluator.Evaluator._Evaluator__calc_APs）
    # 也就是MSCOCO的AP计算中AP、AP50、AP75等的数字的含义
    "MULTI_SCALE_TEST":     False,
    "FLIP_TEST":            False}

