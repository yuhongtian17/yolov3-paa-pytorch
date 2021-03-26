import random
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import argparse
# from tensorboardX import SummaryWriter

import config.yolov3_config as cfg
from eval.evaluator import *
from model.yolov3 import Model_yolov3
from model.loss.yolo_loss import Loss_yolov3
from utils.datasets import UnivDataset
from utils.gpu import *
from utils.tools import *


class Trainer(object):
    """
    Trainer
    def __init__ args:
        gpu_id: int = 0, gpu id
        num_workers: int = 0, number of workers
        weight_path: default = None, the file path of weight or None
        mode: default = None, "+voc" or "+coco" or None
        resume: bool = False, resume training flag
    """
    def __init__(self, gpu_id = 0, num_workers = 0, weight_path = None, mode = None, resume = False):
        self.gpu_id = gpu_id
        self.num_workers = num_workers
        self.weight_path = weight_path
        self.mode = mode
        self.resume = resume

        init_seeds(0)

        if self.mode == "+voc":
            self.__classes = cfg.VOC_CLASS["CLASSES"]
            self.__num_classes = cfg.VOC_CLASS["NUM_CLASSES"]
        else:
            self.__classes = cfg.COCO_CLASS["CLASSES"]
            self.__num_classes = cfg.COCO_CLASS["NUM_CLASSES"]

        self.__anchors = cfg.MODEL["ANCHORS"]
        self.__strides = cfg.MODEL["STRIDES"]

        self.__train_img_size = cfg.TRAIN["TRAIN_IMG_SIZE"]
        self.__batch_size = cfg.TRAIN["BATCH_SIZE"]
        self.__epochs = cfg.TRAIN["EPOCHS"]
        # self.__lr_decay_epochs = cfg.TRAIN["LR_DECAY_EPOCHS"]
        self.__warmup_epochs = cfg.TRAIN["WARMUP_EPOCHS"]
        self.__momentum = cfg.TRAIN["MOMENTUM"]
        self.__weight_decay = cfg.TRAIN["WEIGHT_DECAY"]
        self.__lr_init = cfg.TRAIN["LR_INIT"]
        self.__lr_end = cfg.TRAIN["LR_END"]
        self.__iou_threshold_loss = cfg.TRAIN["IOU_THRESHOLD_LOSS"]
        self.__multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        # self.__augment = cfg.TRAIN["AUGMENT"]

        self.__device = select_device(self.gpu_id)
        self.__train_dataset = UnivDataset(mode = self.mode, img_size = self.__train_img_size, num_classes = self.__num_classes)
        self.__train_dataloader = DataLoader(dataset = self.__train_dataset, batch_size = self.__batch_size, num_workers = self.num_workers, shuffle = True)
        self.__batches = len(self.__train_dataloader)
        self.__start_epoch = 0
        self.__best_mAP = 0.

        self.__model = Model_yolov3(num_classes = self.__num_classes).to(self.__device)
        self.__optimizer = optim.SGD(self.__model.parameters(), lr = self.__lr_init, momentum = self.__momentum, weight_decay = self.__weight_decay)
        self.__criterion = Loss_yolov3(strides = self.__strides, iou_threshold_loss = self.__iou_threshold_loss)

        self.__load_model_weights(self.weight_path, self.resume)
        # self.__model = nn.DataParallel(self.__model, [0, 1, 2, 3])


    def __load_model_weights(self, weight_path, resume):
        print("loading weight file from: {}".format(weight_path))

        # 如果是断点续接
        if resume:
            last_weight = os.path.join(os.path.split(weight_path)[0], "last.pth")
            chkpt = torch.load(last_weight, map_location = self.__device)
            self.__start_epoch = chkpt["epoch"] + 1
            self.__best_mAP = chkpt["best_mAP"]
            self.__model.load_state_dict(chkpt["model"])
            if chkpt["optimizer"] is not None:
                self.__optimizer.load_state_dict(chkpt["optimizer"])
            del chkpt
        # 或者直接加载
        else:
            self.__model.load_darknet_weights(weight_path)

        print("loading weight file is done")


    def __save_model_weights(self, epoch, mAP):
        best_weight = os.path.join(os.path.split(self.weight_path)[0], "best.pth")
        last_weight = os.path.join(os.path.split(self.weight_path)[0], "last.pth")
        ev10_weight = os.path.join(os.path.split(self.weight_path)[0], "backup_epoch{}.pth".format(epoch + 1))
        if mAP > self.__best_mAP:
            self.__best_mAP = mAP
            torch.save(self.__model.state_dict(), best_weight)
        print("best mAP: {}".format(self.__best_mAP))
        chkpt = {
            "epoch": epoch,
            "best_mAP": self.__best_mAP,
            "model": self.__model.state_dict(),
            "optimizer": self.__optimizer.state_dict()}
        torch.save(chkpt, last_weight)

        if (epoch + 1) % 10 == 0:
            torch.save(chkpt, ev10_weight)
        del chkpt


    def train(self):
        print("===== " * 4 + "training " + self.mode + " =====" * 4)

        t1 = datetime.datetime.now()

        for epoch in range(self.__start_epoch, self.__epochs):
            self.__model.train()

            mloss = 0
            for i, (imgs, index_s, index_m, index_l, label_s, label_m, label_l, bboxes) in enumerate(self.__train_dataloader):

                if epoch < self.__warmup_epochs:
                    new_lr = (epoch * self.__batches + i + 1) / (self.__warmup_epochs * self.__batches) * self.__lr_init
                else:
                    # y = cos(x), x ∈ [0, π], y ∈ [1, -1]
                    new_lr = ((np.cos(((epoch - self.__warmup_epochs) * self.__batches + i + 1) # 当前过了多少个batch
                                    / ((self.__epochs - self.__warmup_epochs) * self.__batches) # 除以总的batch数
                                    * np.pi) + 1) * 0.5                                         # 乘以pi加上1乘以0.5得到纵轴上的归一化结果
                              * (self.__lr_init - self.__lr_end) + self.__lr_end)               # 放缩
                for param_group in self.__optimizer.param_groups:
                    param_group["lr"] = new_lr

                imgs = imgs.to(self.__device)
                index_s = index_s.to(self.__device)
                index_m = index_m.to(self.__device)
                index_l = index_l.to(self.__device)
                label_s = label_s.to(self.__device)
                label_m = label_m.to(self.__device)
                label_l = label_l.to(self.__device)
                bboxes = bboxes.to(self.__device)

                p, p_d = self.__model(imgs)

                loss, loss_iou, loss_conf, loss_cls = self.__criterion(p, p_d, index_s, index_m, index_l, label_s, label_m, label_l, bboxes)

                self.__optimizer.zero_grad()
                loss.backward()
                self.__optimizer.step()

                mloss = (mloss * i + loss) / (i + 1)

                # every 10 batches
                if (i + 1) % 10 == 0:
                    # print results
                    t2 = datetime.datetime.now()
                    print("Epoch: {} / {}; Batch: {} / {}; loss: {:.4f}; lr: {:g}; time: {}".format(
                        epoch + 1, self.__epochs, i + 1, self.__batches, mloss, self.__optimizer.param_groups[0]["lr"], t2 - t1))

                    # multi-sclae training (320-608 pixels)
                    if self.__multi_scale_train:
                        self.__train_dataset.img_size = random.choice(range(10, 20)) * 32
                        print("multi_scale_img_size: {}".format(self.__train_dataset.img_size))

            mAP = 0.

            if epoch >= 20: # 训练初期模型很差时测试可能报错
                if self.mode == "+voc":
                    print("===== " * 4 + "evaluating " + self.mode + " =====" * 4)

                    evaluator_voc = Evaluator(self.__model, self.__classes, self.__num_classes, visual = True)
                    APs = evaluator_voc.APs_voc(use_tqdm = False)

                    APsum = 0
                    for i in APs:
                        print("{:16}: {}".format(i, APs[i]))
                        APsum += APs[i]
                    mAP = APsum / self.__num_classes
                    print("mAP: {}".format(mAP))

                else:
                    print("===== " * 4 + "evaluating " + self.mode + " =====" * 4)

                    evaluator_coco = Evaluator(self.__model, self.__classes, self.__num_classes, visual = True)
                    APs = evaluator_coco.APs_coco(use_tqdm = False)

                    mAP = APs[1]
                    print("mAP: {}".format(mAP))

            self.__save_model_weights(epoch, mAP)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type = int, default = 0, help = "gpu id")
    parser.add_argument("--num_workers", type = int, default = 0, help = "number of workers")
    parser.add_argument("--weight_path", type = str, default = "./weight/darknet53_448.weights", help = "the file path of weight or None")
    parser.add_argument("--mode", type = str, default = "+voc", help = "\"+voc\" or \"+coco\" or None")
    parser.add_argument("--resume", type = bool, default = False, help = "resume training flag")
    opt = parser.parse_args()

    Trainer(gpu_id = opt.gpu_id, num_workers = opt.num_workers, weight_path = opt.weight_path, mode = opt.mode, resume = opt.resume).train()
    # Trainer(gpu_id = opt.gpu_id, num_workers = opt.num_workers, weight_path = "./weight/last.pth", mode = opt.mode, resume = True).train()

