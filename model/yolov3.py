# coding=utf-8

import sys
sys.path.append("..")

import torch.nn as nn
import torch
import numpy as np

import config.yolov3_config as cfg
from model.backbones.darknet53 import Darknet53
from model.backbones.cspdarknet53 import CSPDarknet53
from model.head.yolo_head import Head_yolov3
from model.layers.blocks_module import ConvBlock
from model.necks.yolo_fpn import FPN_yolov3, PAN_yolov3
from utils.tools import *


class Model_yolov3(nn.Module):
    """
    Model_yolov3
    def __init__ args:
        num_classes: int
        init_weights: bool = True
    def forward returns:
        p, p_d (p_d is different with yolov3.train() and yolov3.eval())
    """
    def __init__(self, num_classes, init_weights = True):
        super(Model_yolov3, self).__init__()

        self.__num_classes = num_classes
        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__out_channels = cfg.MODEL["ANCHORS_PER_SCLAE"] * (self.__num_classes + 5)

        self.__backnone = Darknet53()
        # print(self.__backnone)
        self.__fpn = PAN_yolov3(filters_in = [256, 512, 1024], filters_out = [self.__out_channels, self.__out_channels, self.__out_channels])

        # small
        self.__head_s = Head_yolov3(num_classes = self.__num_classes, anchors = self.__anchors[0], stride = self.__strides[0])
        # medium
        self.__head_m = Head_yolov3(num_classes = self.__num_classes, anchors = self.__anchors[1], stride = self.__strides[1])
        # large
        self.__head_l = Head_yolov3(num_classes = self.__num_classes, anchors = self.__anchors[2], stride = self.__strides[2])

        # 初始化网络权重
        if init_weights: self.__init_weights()


    def forward(self, x):
        x_s, x_m, x_l = self.__backnone(x)
        x_s, x_m, x_l = self.__fpn(x_s, x_m, x_l)

        out = []
        out.append(self.__head_s(x_s))
        out.append(self.__head_m(x_m))
        out.append(self.__head_l(x_l))

        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0) # 把p_d缝合起来


    def __init_weights(self):
        """
        初始化网络权重
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

                # print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

                # print("initing {}".format(m))


    def load_darknet_weights(self, weight_file, cutoff = 52):
        # https://github.com/ultralytics/yolov3/blob/master/models.py

        with open(weight_file, "rb") as f:
            _ = np.fromfile(f, dtype = np.int32, count = 5)
            weights = np.fromfile(f, dtype = np.float32)

        count = 0
        ptr = 0
        for m in self.modules():
            if isinstance(m, ConvBlock):
                # only initing backbone conv's weights
                # cutoff:
                # Darknet: 52
                # CSPDarkNet: 72
                if count == cutoff:
                    break
                count += 1
                # print(count, m)

                conv_layer = m._ConvBlock__conv_layer
                if m.norm == "bn":
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = m._ConvBlock__norm_layer
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

