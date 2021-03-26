# coding=utf-8

import sys
sys.path.append("../utils")

import torch
import torch.nn as nn

import sklearn.mixture as skm

import config.yolov3_config as cfg
from utils import tools


class FocalLoss(nn.Module):
    # https://arxiv.org/abs/1708.02002
    def __init__(self, gamma = 2.0, alpha = 1.0, reduction = "mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction = reduction)

    def forward(self, input, target):
        loss = self.__loss(input = input, target = target)
        loss *= self.__alpha * torch.pow(torch.abs(target - torch.sigmoid(input)), self.__gamma)
        return loss


class Loss_yolov3(nn.Module):
    def __init__(self, strides, iou_threshold_loss = 0.5):
        super(Loss_yolov3, self).__init__()
        self.__strides = strides

        self.__batch_size = None
        self.__device = None

    def forward(self, p, p_d, index_s, index_m, index_l, label_s, label_m, label_l, bboxes):
        """
        """
        self.__batch_size = p[0].size(0)
        self.__device = p[0].device
        label_obj_mask_s, label_obj_mask_m, label_obj_mask_l = self.__paa(p, p_d, index_s, index_m, index_l, label_s, label_m, label_l, bboxes)

        loss_s, loss_iou_s, loss_conf_s, loss_cls_s = self.__cal_loss_per_layer(p[0], p_d[0], label_s, label_obj_mask_s, self.__strides[0])
        loss_m, loss_iou_m, loss_conf_m, loss_cls_m = self.__cal_loss_per_layer(p[1], p_d[1], label_m, label_obj_mask_m, self.__strides[1])
        loss_l, loss_iou_l, loss_conf_l, loss_cls_l = self.__cal_loss_per_layer(p[2], p_d[2], label_l, label_obj_mask_l, self.__strides[2])

        loss = loss_l + loss_m + loss_s
        loss_iou = loss_iou_s + loss_iou_m + loss_iou_l
        loss_conf = loss_conf_s + loss_conf_m + loss_conf_l
        loss_cls = loss_cls_s + loss_cls_m + loss_cls_l

        return loss, loss_iou, loss_conf, loss_cls

    def __paa(self, p, p_d, index_s, index_m, index_l, label_s, label_m, label_l, bboxes):
        # 计算PAA方式Loss
        num_bboxes = bboxes.size(1)

        loss_s = self.__cal_loss_per_layer_for_paa(p[0], p_d[0], label_s, self.__strides[0])
        loss_m = self.__cal_loss_per_layer_for_paa(p[1], p_d[1], label_m, self.__strides[1])
        loss_l = self.__cal_loss_per_layer_for_paa(p[2], p_d[2], label_l, self.__strides[2])

        # 全False的初始化的mask
        label_obj_mask_s = loss_s < 0
        label_obj_mask_m = loss_m < 0
        label_obj_mask_l = loss_l < 0

        # 为了统计mask的计数是否正确
        loss_min_K_s_sum = 0
        loss_min_K_m_sum = 0
        loss_min_K_l_sum = 0
        aaaa_sum = 0

        for i in range(self.__batch_size):
            for j in range(num_bboxes):
                # 挑选出负责第i张图第j个bbox的那些锚框
                index_mask_s = (index_s[i] == j)
                index_mask_m = (index_m[i] == j)
                index_mask_l = (index_l[i] == j)

                # 初筛
                if index_mask_s.sum() == 0 and index_mask_m.sum() == 0 and index_mask_l.sum() == 0:
                    continue

                # 每层挑选出最小的K个
                loss_min_K_s, K_thresh_s = self.__min_K(loss_s[i][index_mask_s])
                loss_min_K_m, K_thresh_m = self.__min_K(loss_m[i][index_mask_m])
                loss_min_K_l, K_thresh_l = self.__min_K(loss_l[i][index_mask_l])

                loss_min_K_s_sum += loss_min_K_s.size(0)
                loss_min_K_m_sum += loss_min_K_m.size(0)
                loss_min_K_l_sum += loss_min_K_l.size(0)

                # 3K个合在一起进行GMM，得到阈值
                loss_thresh, aaaa = self.__gmm(loss_min_K_s, loss_min_K_m, loss_min_K_l)
                aaaa_sum += aaaa

                # 确定最终阈值
                loss_thresh_s = min(loss_thresh, K_thresh_s)
                loss_thresh_m = min(loss_thresh, K_thresh_m)
                loss_thresh_l = min(loss_thresh, K_thresh_l)

                # 创建与写入mask
                loss_mask_s = (loss_s[i] <= loss_thresh_s)
                loss_mask_m = (loss_m[i] <= loss_thresh_m)
                loss_mask_l = (loss_l[i] <= loss_thresh_l)

                obj_mask_s = index_mask_s.unsqueeze(-1) & loss_mask_s
                obj_mask_m = index_mask_m.unsqueeze(-1) & loss_mask_m
                obj_mask_l = index_mask_l.unsqueeze(-1) & loss_mask_l

                label_obj_mask_s[i] |= obj_mask_s
                label_obj_mask_m[i] |= obj_mask_m
                label_obj_mask_l[i] |= obj_mask_l

        print("printpoint 1:", loss_min_K_s_sum, loss_min_K_m_sum, loss_min_K_l_sum, aaaa_sum)
        print("printpoint 2:", label_obj_mask_s.sum(), label_obj_mask_m.sum(), label_obj_mask_l.sum(), label_obj_mask_s.sum() + label_obj_mask_m.sum() + label_obj_mask_l.sum())

        label_obj_mask_s = label_obj_mask_s.type(torch.FloatTensor).detach().to(self.__device)
        label_obj_mask_m = label_obj_mask_m.type(torch.FloatTensor).detach().to(self.__device)
        label_obj_mask_l = label_obj_mask_l.type(torch.FloatTensor).detach().to(self.__device)

        #index_i_s = self.__index_with_batch_size(index_s).type(torch.LongTensor)
        #index_j_s = index_s.type(torch.LongTensor)
        #index_i_m = self.__index_with_batch_size(index_m).type(torch.LongTensor)
        #index_j_m = index_m.type(torch.LongTensor)
        #index_i_l = self.__index_with_batch_size(index_l).type(torch.LongTensor)
        #index_j_l = index_l.type(torch.LongTensor)
        ## print("line 96:", index_i_s.size(), index_j_s.size(), index_i_m.size(), index_j_m.size(), index_i_l.size(), index_j_l.size())

        #loss_thresh_s = loss_thresh_all[index_i_s, index_j_s].unsqueeze(-1)
        #loss_thresh_m = loss_thresh_all[index_i_m, index_j_m].unsqueeze(-1)
        #loss_thresh_l = loss_thresh_all[index_i_l, index_j_l].unsqueeze(-1)
        ## print("line 101:", loss_thresh_s.size(), loss_thresh_m.size(), loss_thresh_l.size())

        #label_obj_mask_s2 = (loss_s <= loss_thresh_s).type(torch.FloatTensor).detach().to(self.__device)
        #label_obj_mask_m2 = (loss_m <= loss_thresh_m).type(torch.FloatTensor).detach().to(self.__device)
        #label_obj_mask_l2 = (loss_l <= loss_thresh_l).type(torch.FloatTensor).detach().to(self.__device)
        #print("line 106:", label_obj_mask_s2.sum(), label_obj_mask_m2.sum(), label_obj_mask_l2.sum(), label_obj_mask_s2.sum() + label_obj_mask_m2.sum() + label_obj_mask_l2.sum())

        return label_obj_mask_s, label_obj_mask_m, label_obj_mask_l

    def __cal_loss_per_layer_for_paa(self, p, p_d, label, stride, lambda_cls = 1.0, lambda_iou = 1.0, lambda_conf = 0.5):
        """
        outputs: torch.size([batch_size, num_grids(h), num_grids(w), num_anchors, 1])
        """
        BCE = nn.BCEWithLogitsLoss(reduction = "none")
        FOCAL = FocalLoss(gamma = 2, alpha = 1.0, reduction = "none")

        # p: torch.size([batch_size, num_grids(h), num_grids(w), num_anchors, (tx, ty, tw, th, to, num_classes)])
        batch_size, num_grids = p.shape[:2]
        img_size = num_grids * stride

        pred_xywh = p_d[..., 0:4] # 取实际值
        pred_conf = p[..., 4:5] # 取未经sigmoid的
        pred_cls = p[..., 5:] # 取未经sigmoid的

        label_xywh = label[..., 0:4]
        # label_obj_mask = label[..., 4:5]
        label_mix = label[..., 5:6]
        label_cls = label[..., 6:]

        # ##### loss iou #####
        iou = tools.ciou_xywh_torch(pred_xywh, label_xywh).unsqueeze(-1)

        # The scaled weight of bbox is used to balance the impact of small objects and large objects on loss.
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (img_size ** 2)
        loss_iou = bbox_loss_scale * (1.0 - iou) * label_mix * lambda_iou

        # ##### loss confidence #####
        # iou_conf = tools.iou_xywh_torch(pred_xywh, label_xywh).unsqueeze(-1)
        loss_conf = FOCAL(input = pred_conf, target = pred_conf * 0 + 1) * label_mix * lambda_conf

        # ##### loss classes #####
        loss_cls = BCE(input = pred_cls, target = label_cls) * label_mix * lambda_cls
        loss_cls = torch.sum(loss_cls, dim = -1, keepdim = True)

        loss = loss_iou + loss_conf + loss_cls
        return loss

    def __min_K(self, loss, K = 15, KK = 15):
        """
        get loss min K anchors
        args:
            loss: torch.size([num_anchors, 1])
        returns:
            min K values
            thresh in min K values
        """
        num_anchors, temp = loss.size()
        # assert temp == 1
        # print("line 155 min:", loss.min())
        if num_anchors == 0:
            return loss, 0
        elif num_anchors <= KK:
            return loss, loss.max()
        elif num_anchors <= K:
            values, indices = torch.sort(loss, dim = 0)
            return values, values[KK - 1]
        else:
            values, indices = torch.topk(loss, K, dim = 0, largest = False)
            return values, values[KK - 1]

    def __gmm(self, loss_s, loss_m, loss_l):
        """
        Gaussian Mixture Model
        args:
            loss_s, loss_m, loss_l: torch.size([num_anchors, 1])
        returns:
            thresh
            number of value <= thresh
        """
        device = self.__device
        candidate_loss = torch.cat([loss_s, loss_m, loss_l], dim = 0)
        candidate_num = candidate_loss.size(0)
        if candidate_num > 1:
            candidate_loss, inds = torch.sort(candidate_loss, dim = 0)
            candidate_loss = candidate_loss.view(-1, 1).cpu().detach().numpy()
            min_loss, max_loss = candidate_loss.min(), candidate_loss.max()
            means_init=[[min_loss], [max_loss]]
            weights_init = [0.5, 0.5]
            precisions_init=[[[1.0]], [[1.0]]]
            gmm = skm.GaussianMixture(2,
                                        weights_init=weights_init,
                                        means_init=means_init,
                                        precisions_init=precisions_init)
            gmm.fit(candidate_loss)
            components = gmm.predict(candidate_loss)
            scores = gmm.score_samples(candidate_loss)
            components = torch.from_numpy(components).to(device)
            scores = torch.from_numpy(scores).to(device)
            fgs = components == 0
            bgs = components == 1
            if torch.nonzero(fgs, as_tuple=False).numel() > 0:
                # Fig 3. (c)
                fg_max_score = scores[fgs].max().item()
                fg_max_idx = torch.nonzero(fgs & (scores == fg_max_score), as_tuple=False).min()
                thresh = candidate_loss[fg_max_idx]
            else:
                thresh = candidate_loss[0]
                fg_max_idx = 0
        elif candidate_num == 1:
            thresh = candidate_loss[0]
            fg_max_idx = 0
        else:
            thresh = 0
            fg_max_idx = -1

        return float(thresh), int(fg_max_idx + 1)

    def __index_with_batch_size(self, index):
        # index: torch.size([batch_size, num_grids(h), num_grids(w), num_anchors])
        batch_size, num_grids_h, num_grids_w, num_anchors = index.size()
        index_batch = torch.arange(0, batch_size).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, num_grids_h, num_grids_w, num_anchors)
        return index_batch

    def __cal_loss_per_layer(self, p, p_d, label, label_obj_mask, stride):
        """
        """
        BCE = nn.BCEWithLogitsLoss(reduction = "none")
        FOCAL = FocalLoss(gamma = 2, alpha = 1.0, reduction = "none")

        # p: torch.size([batch_size, num_grids(h), num_grids(w), num_anchors, (tx, ty, tw, th, to, num_classes)])
        batch_size, num_grids = p.shape[:2]
        img_size = num_grids * stride

        pred_xywh = p_d[..., 0:4] # 取实际值
        pred_conf = p[..., 4:5] # 取未经sigmoid的
        pred_cls = p[..., 5:] # 取未经sigmoid的

        label_xywh = label[..., 0:4]
        # label_obj_mask = label[..., 4:5]
        label_mix = label[..., 5:6]
        label_cls = label[..., 6:]

        # ##### loss iou #####
        iou = tools.ciou_xywh_torch(pred_xywh, label_xywh).unsqueeze(-1)

        # The scaled weight of bbox is used to balance the impact of small objects and large objects on loss.
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (img_size ** 2)
        loss_iou = label_obj_mask * bbox_loss_scale * (1.0 - iou) * label_mix

        # ##### loss confidence #####
        loss_conf = FOCAL(input = pred_conf, target = label_obj_mask) * label_mix

        # ##### loss classes #####
        loss_cls = label_obj_mask * BCE(input = pred_cls, target = label_cls) * label_mix

        loss_iou = (torch.sum(loss_iou)) / batch_size
        loss_conf = (torch.sum(loss_conf)) / batch_size
        loss_cls = (torch.sum(loss_cls)) / batch_size
        loss = loss_iou + loss_conf + loss_cls

        return loss, loss_iou, loss_conf, loss_cls

