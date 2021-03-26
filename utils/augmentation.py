# coding=utf-8
# reference of v2: https://github.com/libo-coder/yolo_v4_pytorch3/blob/main/utils/datasets.py

import random
import math

import torch
import numpy as np
import cv2


class RandomHSV_v2(object):
    def __init__(self, hgain = 0.5, sgain = 0.5, vgain = 0.5, p = 0.5):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=np.int16)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

        return img


class RandomAffine_v2(object):
    def __init__(self, degrees = 10, translate = 0.1, scale = 0.1, shear = 10, border = 0, p = 0.5):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.border = border
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            height = img.shape[0] + self.border * 2
            width = img.shape[1] + self.border * 2

            # Rotation and Scale
            R = np.eye(3)
            a = random.uniform(-self.degrees, self.degrees)
            # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
            s = random.uniform(1 - self.scale, 1 + self.scale)
            R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

            # Translation
            T = np.eye(3)
            T[0, 2] = random.uniform(-self.translate, self.translate) * img.shape[0] + self.border  # x translation (pixels)
            T[1, 2] = random.uniform(-self.translate, self.translate) * img.shape[1] + self.border  # y translation (pixels)

            # Shear
            S = np.eye(3)
            S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
            S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

            # Combined rotation matrix
            M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
            if (self.border != 0) or (M != np.eye(3)).any():  # image changed
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

            # Transform label coordinates
            n = len(bboxes)
            if n:
                # warp points
                xy = np.ones((n * 4, 3))
                xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                xy = (xy @ M.T)[:, :2].reshape(n, 8)

                # create new boxes
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                # # apply angle-based reduction of bounding boxes
                # radians = a * math.pi / 180
                # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
                # x = (xy[:, 2] + xy[:, 0]) / 2
                # y = (xy[:, 3] + xy[:, 1]) / 2
                # w = (xy[:, 2] - xy[:, 0]) * reduction
                # h = (xy[:, 3] - xy[:, 1]) * reduction
                # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

                # reject warped points outside of image
                xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
                xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

                bboxes[:, 0:4] = xy[:, 0:4]

        return img, bboxes


class Mosaic_v2(object):
    def __init__(self, img_size, p = 0.5):
        self.img_size = img_size
        self.p = p

    def __call__(self, img_list, bboxes_list):
        if random.random() < self.p:
            s = self.img_size
            xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y

            img4 = np.full((s * 2, s * 2, img_list[0].shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            bboxes4 = []

            for i in range(4):
                img = img_list[i]
                h, w, _ = img.shape

                # place img in img4
                if i == 0:  # top left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

                img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
                padw = x1a - x1b
                padh = y1a - y1b

                bboxes = bboxes_list[i]
                if bboxes.size > 0:
                    bboxes = self.__crop_bboxes(bboxes, x1b, y1b, x2b, y2b)
                    bboxes[:, 0:4] = bboxes[:, 0:4] + np.array([padw, padh, padw, padh])
                bboxes4.append(bboxes)

            bboxes4 = np.concatenate(bboxes4, 0)

        else:
            img4 = img_list[0]
            bboxes4 = bboxes_list[0]

        return img4, bboxes4

    def __crop_bboxes(self, bboxes, xmin, ymin, xmax, ymax):
        bboxes = np.concatenate([np.maximum(bboxes[:, 0:2], [xmin, ymin]),
                                 np.minimum(bboxes[:, 2:4], [xmax, ymax]),
                                 bboxes[:, 4:]], axis = -1)
        return bboxes


class RandomHorizontalFilp(object):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            _, w_img, _ = img.shape
            # img = np.fliplr(img)
            img = img[:, ::-1, :]
            bboxes[:, [0, 2]] = w_img - bboxes[:, [2, 0]]

        return img, bboxes


class RandomCrop(object):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img, bboxes):
        # random.random() 方法返回随机生成的一个实数，它在 [0, 1) 范围内。
        # random.uniform() 方法返回随机生成的一个实数，它在 [x, y) 范围内。
        if random.random() < self.p:
            h_img, w_img, _ = img.shape

            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis = 0), np.max(bboxes[:, 2:4], axis = 0)], axis = -1)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = min(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = min(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            img = img[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return img, bboxes


class Resize(object):
    """
    调整图片大小
    __init__ args:
        target_shape: (h_target, w_target)，调整后的图片大小
        correct_box: bool = False，对框也进行对应调整
    __call__ args:
        img: 待调整的图片
        bboxes: default = None，待调整的框，实际值
    returns:
        image: 调整后的图片
        bboxes: 调整后的框（如果correct_box == True），实际值
    notes:
        将图片转为目标大小，BGR转换为RGB，归一化到[0, 1]上
        bboxes依然是以图片大小为参考的实际值
    """
    def __init__(self, target_shape, correct_box = False):
        self.h_target, self.w_target = target_shape
        self.correct_box = correct_box

    def __call__(self, img, bboxes = None):
        h_org, w_org, _= img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        resize_ratio = min(1.0 * self.w_target / w_org, 1.0 * self.h_target / h_org)
        resize_w = max(int(resize_ratio * w_org), 1) # 防止放缩后为0
        resize_h = max(int(resize_ratio * h_org), 1) # 防止放缩后为0
        image_resized = cv2.resize(img, (resize_w, resize_h))

        image_paded = np.full((self.h_target, self.w_target, 3), 128.0) # 填充而非拉伸
        dw = int((self.w_target - resize_w) / 2)
        dh = int((self.h_target - resize_h) / 2)
        image_paded[dh:resize_h + dh, dw:resize_w + dw, :] = image_resized
        image = image_paded / 255.0  # normalize to [0, 1]

        if self.correct_box:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh
            return image, bboxes
        return image


class Mixup(object):
    """
    Mixup: Beyond Empirical Risk Minimization
    # https://arxiv.org/abs/1710.09412
    """
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img_org, bboxes_org, img_mix, bboxes_mix):
        if random.random() < self.p:
            lam = np.random.beta(1.5, 1.5)
            img = lam * img_org + (1 - lam) * img_mix
            bboxes_org = np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), lam)], axis = 1) # mix = lam
            bboxes_mix = np.concatenate([bboxes_mix, np.full((len(bboxes_mix), 1), 1.0 - lam)], axis = 1) # mix = (1.0 - lam)
            bboxes = np.concatenate([bboxes_org, bboxes_mix])
        else:
            img = img_org
            bboxes = np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis = 1) # mix = 1.0

        return img, bboxes

