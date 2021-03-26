# YOLOv3-PAA with PyTorch, PASCAL VOC and MS COCO

This is a repository for my Graduation Project of UCAS.

YOLOv3 baseline > https://github.com/AkitsukiM/YOLOv3x-pytorch

PAA > https://github.com/kkhoot/PAA

## How to use it

same as baseline > https://github.com/AkitsukiM/YOLOv3x-pytorch/blob/main/README.md

## What's different

./utils/datasets.py
./model/loss/yolo_loss.py
./model/head/yolo_head.py
./train.py

## mAP on PASCAL VOC

YOLOv3                    : 0.8369
YOLOv3+RetinaNet          : 0.8334
YOLOv3          +PAA 15-5 : 0.8310

YOLOv3+RetinaNet+PAA 15-15: 0.8352
                     15-10: 0.8357
                     15-5 : 0.8391 (best)
                     15-3 : 0.8348
                     10-10: 0.8341
                     10-5 : 0.8364
                     10-3 : 0.8379

## AP on MS COCO

つづく

-----

Copyright (c) 2021 HONGTIAN YU. All rights reserved.

Date modified: 2021/03/26

