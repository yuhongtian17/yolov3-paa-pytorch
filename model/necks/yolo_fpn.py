# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.blocks_module import ConvBlock


class Upsample(nn.Module):
    def __init__(self, scale_factor = 1, mode = "nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor = self.scale_factor, mode = self.mode)


class Route(nn.Module):
    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x1, x2):
        out = torch.cat((x2, x1), dim = 1)
        return out


class FPN_yolov3(nn.Module):
    """
    FPN for yolov3, and is different from original FPN or retinanet FPN.
    """
    def __init__(self, filters_in, filters_out):
        super(FPN_yolov3, self).__init__()

        fi_s, fi_m, fi_l = filters_in
        fo_s, fo_m, fo_l = filters_out

        # large的特征图
        self.__conv_l_set = nn.Sequential(
            ConvBlock(fi_l, 512, kernel_size = 1, norm = "bn", activate = "leaky"),
            ConvBlock(512, 1024, kernel_size = 3, norm = "bn", activate = "leaky"),
            ConvBlock(1024, 512, kernel_size = 1, norm = "bn", activate = "leaky"),
            ConvBlock(512, 1024, kernel_size = 3, norm = "bn", activate = "leaky"),
            ConvBlock(1024, 512, kernel_size = 1, norm = "bn", activate = "leaky"))
        # large的上传支路
        self.__conv_l_up = ConvBlock(512, 256, kernel_size = 1, norm = "bn", activate = "leaky")
        self.__upsample_l = Upsample(scale_factor = 2)
        self.__route_l = Route()

        # medium的特征图
        self.__conv_m_set = nn.Sequential(
            ConvBlock(fi_m + 256, 256, kernel_size = 1, norm = "bn", activate = "leaky"),
            ConvBlock(256, 512, kernel_size = 3, norm = "bn", activate = "leaky"),
            ConvBlock(512, 256, kernel_size = 1, norm = "bn", activate = "leaky"),
            ConvBlock(256, 512, kernel_size = 3, norm = "bn", activate = "leaky"),
            ConvBlock(512, 256, kernel_size = 1, norm = "bn", activate = "leaky"))
        # medium的上传支路
        self.__conv_m_up = ConvBlock(256, 128, kernel_size = 1, norm = "bn", activate = "leaky")
        self.__upsample_m = Upsample(scale_factor = 2)
        self.__route_m = Route()

        # small的特征图
        self.__conv_s_set = nn.Sequential(
            ConvBlock(fi_s + 128, 128, kernel_size = 1, norm = "bn", activate = "leaky"),
            ConvBlock(128, 256, kernel_size = 3, norm = "bn", activate = "leaky"),
            ConvBlock(256, 128, kernel_size = 1, norm = "bn", activate = "leaky"),
            ConvBlock(128, 256, kernel_size = 3, norm = "bn", activate = "leaky"),
            ConvBlock(256, 128, kernel_size = 1, norm = "bn", activate = "leaky"))

        # large的输出支路
        self.__conv_l_out0 = ConvBlock(512, 1024, kernel_size = 3, norm = "bn", activate = "leaky")
        self.__conv_l_out1 = ConvBlock(1024, fo_l, kernel_size = 1) # 没有bn和relu
        # medium的输出支路
        self.__conv_m_out0 = ConvBlock(256, 512, kernel_size = 3, norm = "bn", activate = "leaky")
        self.__conv_m_out1 = ConvBlock(512, fo_m, kernel_size = 1) # 没有bn和relu
        # small的输出支路
        self.__conv_s_out0 = ConvBlock(128, 256, kernel_size = 3, norm = "bn", activate = "leaky")
        self.__conv_s_out1 = ConvBlock(256, fo_s, kernel_size = 1) # 没有bn和relu

    def forward(self, x_s, x_m, x_l):
        """
        """
        # large的特征图
        p_l = self.__conv_l_set(x_l)
        # large的上传支路
        r_l = self.__conv_l_up(p_l)
        r_l = self.__upsample_l(r_l)
        x_m = self.__route_l(x_m, r_l)

        # medium的特征图
        p_m = self.__conv_m_set(x_m)
        # medium的上传支路
        r_m = self.__conv_m_up(p_m)
        r_m = self.__upsample_m(r_m)
        x_s = self.__route_m(x_s, r_m)

        # small的特征图
        p_s = self.__conv_s_set(x_s)

        # large的输出支路
        o_l = self.__conv_l_out0(p_l)
        o_l = self.__conv_l_out1(o_l)
        # medium的输出支路
        o_m = self.__conv_m_out0(p_m)
        o_m = self.__conv_m_out1(o_m)
        # small的输出支路
        o_s = self.__conv_s_out0(p_s)
        o_s = self.__conv_s_out1(o_s)

        return o_s, o_m, o_l


class PAN_yolov3(nn.Module):
    """
    """
    def __init__(self, filters_in, filters_out):
        super(PAN_yolov3, self).__init__()

        fi_s, fi_m, fi_l = filters_in
        fo_s, fo_m, fo_l = filters_out

        # large的特征图
        self.__conv_l_set = nn.Sequential(
            ConvBlock(fi_l, 512, kernel_size = 1, norm = "bn", activate = "leaky"),
            ConvBlock(512, 1024, kernel_size = 3, norm = "bn", activate = "leaky"),
            ConvBlock(1024, 512, kernel_size = 1, norm = "bn", activate = "leaky"),
            ConvBlock(512, 1024, kernel_size = 3, norm = "bn", activate = "leaky"),
            ConvBlock(1024, 512, kernel_size = 1, norm = "bn", activate = "leaky"))
        # large的上传支路
        self.__conv_l_up = ConvBlock(512, 256, kernel_size = 1, norm = "bn", activate = "leaky")
        self.__upsample_l = Upsample(scale_factor = 2)
        self.__route_l = Route()

        # medium的特征图
        self.__conv_m_set = nn.Sequential(
            ConvBlock(fi_m + 256, 256, kernel_size = 1, norm = "bn", activate = "leaky"),
            ConvBlock(256, 512, kernel_size = 3, norm = "bn", activate = "leaky"),
            ConvBlock(512, 256, kernel_size = 1, norm = "bn", activate = "leaky"),
            ConvBlock(256, 512, kernel_size = 3, norm = "bn", activate = "leaky"),
            ConvBlock(512, 256, kernel_size = 1, norm = "bn", activate = "leaky"))
        # medium的上传支路
        self.__conv_m_up = ConvBlock(256, 128, kernel_size = 1, norm = "bn", activate = "leaky")
        self.__upsample_m = Upsample(scale_factor = 2)
        self.__route_m = Route()

        # small的特征图
        self.__conv_s_set = nn.Sequential(
            ConvBlock(fi_s + 128, 128, kernel_size = 1, norm = "bn", activate = "leaky"),
            ConvBlock(128, 256, kernel_size = 3, norm = "bn", activate = "leaky"),
            ConvBlock(256, 128, kernel_size = 1, norm = "bn", activate = "leaky"),
            ConvBlock(128, 256, kernel_size = 3, norm = "bn", activate = "leaky"),
            ConvBlock(256, 128, kernel_size = 1, norm = "bn", activate = "leaky"))

        # small的第二特征图
        # 即small的特征图
        # medium的第二特征图
        self.__conv_s_down0 = ConvBlock(128, 256, kernel_size = 3, stride = 2, norm = "bn", activate = "leaky")
        self.__conv_s_down1 = ConvBlock(256, 256, kernel_size = 3, norm = "bn", activate = "leaky")
        # large的第二特征图
        self.__conv_m_down0 = ConvBlock(256, 512, kernel_size = 3, stride = 2, norm = "bn", activate = "leaky")
        self.__conv_m_down1 = ConvBlock(512, 512, kernel_size = 3, norm = "bn", activate = "leaky")

        # large的输出支路
        self.__conv_l_out0 = ConvBlock(512, 1024, kernel_size = 3, norm = "bn", activate = "leaky")
        self.__conv_l_out1 = ConvBlock(1024, fo_l, kernel_size = 1) # 没有bn和relu
        # medium的输出支路
        self.__conv_m_out0 = ConvBlock(256, 512, kernel_size = 3, norm = "bn", activate = "leaky")
        self.__conv_m_out1 = ConvBlock(512, fo_m, kernel_size = 1) # 没有bn和relu
        # small的输出支路
        self.__conv_s_out0 = ConvBlock(128, 256, kernel_size = 3, norm = "bn", activate = "leaky")
        self.__conv_s_out1 = ConvBlock(256, fo_s, kernel_size = 1) # 没有bn和relu

    def forward(self, x_s, x_m, x_l):
        """
        """
        # large的特征图
        p_l = self.__conv_l_set(x_l)
        # large的上传支路
        r_l = self.__conv_l_up(p_l)
        r_l = self.__upsample_l(r_l)
        x_m = self.__route_l(x_m, r_l)

        # medium的特征图
        p_m = self.__conv_m_set(x_m)
        # medium的上传支路
        r_m = self.__conv_m_up(p_m)
        r_m = self.__upsample_m(r_m)
        x_s = self.__route_m(x_s, r_m)

        # small的特征图
        p_s = self.__conv_s_set(x_s)

        # small的第二特征图
        n_s = p_s
        # medium的第二特征图
        rr_s = self.__conv_s_down0(n_s)
        n_m = p_m + rr_s
        n_m = self.__conv_s_down1(n_m)
        # large的第二特征图
        rr_m = self.__conv_m_down0(n_m)
        n_l = p_l + rr_m
        n_l = self.__conv_m_down1(n_l)

        # large的输出支路
        o_l = self.__conv_l_out0(n_l)
        o_l = self.__conv_l_out1(o_l)
        # medium的输出支路
        o_m = self.__conv_m_out0(n_m)
        o_m = self.__conv_m_out1(o_m)
        # small的输出支路
        o_s = self.__conv_s_out0(n_s)
        o_s = self.__conv_s_out1(o_s)

        return o_s, o_m, o_l

