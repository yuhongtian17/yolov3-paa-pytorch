# coding=utf-8

import torch
import torch.nn as nn


class Head_yolov3(nn.Module):
    def __init__(self, num_classes, anchors, stride):
        super(Head_yolov3, self).__init__()
        self.__num_classes = num_classes
        self.__anchors = anchors
        self.__num_anchors = len(anchors)
        self.__stride = stride

    def forward(self, p):
        # 这里输入的p是网络的单个feature map输出
        batch_size, num_grids = p.shape[0], p.shape[-1]
        # 这里输出的p是重新组织了的单个feature map输出
        p = p.view(batch_size, self.__num_anchors, 5 + self.__num_classes, num_grids, num_grids).permute(0, 3, 4, 1, 2)
        # 这里输出的p_decode是p的解码结果
        p_decode = self.__decode(p.clone())

        return (p, p_decode)

    def __decode(self, p):
        batch_size, num_grids = p.shape[0], p.shape[1]

        device = p.device # 获取所在的设备
        anchors = (1.0 * self.__anchors).to(device) # 把anchor的数据送到设备上去

        # 原始数据
        conv_raw_dxdy = p[:, :, :, :, 0:2]
        conv_raw_dwdh = p[:, :, :, :, 2:4]
        conv_raw_conf = p[:, :, :, :, 4:5]
        conv_raw_prob = p[:, :, :, :, 5:]

        # 创建一个grid索引张量
        y = torch.arange(0, num_grids).unsqueeze(1).repeat(1, num_grids)
        x = torch.arange(0, num_grids).unsqueeze(0).repeat(num_grids, 1)
        grid_xy = torch.stack([x, y], dim = -1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)

        # 解码（解码成以原图大小为标准的值）
        # pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * self.__stride
        pred_xy = (conv_raw_dxdy + grid_xy) * self.__stride
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * self.__stride # 尽管多尺度训练，但锚框大小并不相应变化！
        # pred_xywh = torch.cat([pred_xy, pred_wh], dim = -1)
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xy, pred_wh, pred_conf, pred_prob], dim = -1)

        # 如果不是training mode，那么改成二维列表输出；否则原样输出
        return pred_bbox.view(-1, 5 + self.__num_classes) if not self.training else pred_bbox

