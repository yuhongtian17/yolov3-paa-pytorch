import os
import shutil

import cv2
import argparse
from tqdm import tqdm

import config.yolov3_config as cfg
from eval.evaluator import Evaluator
from model.yolov3 import Model_yolov3
from utils.gpu import *
from utils.tools import *
from utils.visualize import *


class Tester(object):
    """
    Tester
    def __init__ args:
        gpu_id: int = 0, gpu id
        weight_path: default = None, the file path of weight or None
        mode: default = None, the folder path of image or "+voc" or "+coco" or None
    """
    def __init__(self, gpu_id = 0, weight_path = None, mode = None):
        self.gpu_id = gpu_id
        self.weight_path = weight_path
        self.mode = mode

        if self.mode == "+voc":
            self.__classes = cfg.VOC_CLASS["CLASSES"]
            self.__num_classes = cfg.VOC_CLASS["NUM_CLASSES"]
        else:
            self.__classes = cfg.COCO_CLASS["CLASSES"]
            self.__num_classes = cfg.COCO_CLASS["NUM_CLASSES"]

        self.__multi_scale_test = cfg.TEST["MULTI_SCALE_TEST"]
        self.__flip_test = cfg.TEST["FLIP_TEST"]

        self.__device = select_device(self.gpu_id) # 先确定gpu编号
        self.__model = Model_yolov3(num_classes = self.__num_classes).to(self.__device) # 再加载初始模型，并送到设备上
        self.__load_model_weights(self.weight_path) # 再加载训练好的模型


    def __load_model_weights(self, weight_path):
        print("loading weight file from: {}".format(weight_path))

        chkpt = torch.load(weight_path, map_location = self.__device)
        self.__model.load_state_dict(chkpt)
        del chkpt

        print("loading weight file is done")


    def test(self):
        if self.mode not in ["+voc", "+coco", None]:
            print("===== " * 4 + "visualizing " + self.mode + " =====" * 4)

            # 创建新的visual目录
            visual_result_path = os.path.join(cfg.PROJECT_PATH, "data", "visual")
            if os.path.exists(visual_result_path):
                shutil.rmtree(visual_result_path)
            os.mkdir(visual_result_path)

            imgs = os.listdir(self.mode)
            for v in tqdm(imgs):
                path = os.path.join(self.mode, v)
                # print("test images: {}".format(path))

                img = cv2.imread(path)
                assert img is not None

                evaluator_imgs = Evaluator(self.__model, self.__classes, self.__num_classes, visual = False)
                bboxes_prd = evaluator_imgs.get_bbox(img)
                if bboxes_prd.shape[0] != 0:
                    boxes = bboxes_prd[..., :4]
                    class_inds = bboxes_prd[..., 5].astype(np.int32)
                    scores = bboxes_prd[..., 4]

                    visualize_boxes(image = img, boxes = boxes, labels = class_inds, probs = scores, class_labels = self.__classes)
                    path = os.path.join(visual_result_path, v)
                    cv2.imwrite(path, img)
                    # print("saved images: {}".format(path))

        elif self.mode == "+voc":
            print("===== " * 4 + "evaluating " + self.mode + " =====" * 4)

            evaluator_voc = Evaluator(self.__model, self.__classes, self.__num_classes, visual = True)
            APs = evaluator_voc.APs_voc(multi_scale_test = self.__multi_scale_test, flip_test = self.__flip_test)

            APsum = 0
            for i in APs:
                print("{:16}: {}".format(i, APs[i]))
                APsum += APs[i]
            mAP = APsum / self.__num_classes
            print("mAP: {}".format(mAP))

        else:
            print("===== " * 4 + "evaluating " + self.mode + " =====" * 4)

            evaluator_coco = Evaluator(self.__model, self.__classes, self.__num_classes, visual = True)
            APs = evaluator_coco.APs_coco()
            # cocoapi自带print


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type = int, default = 0, help = "gpu id")
    parser.add_argument("--weight_path", type = str, default = "./weight/best.pth", help = "the file path of weight or None")
    parser.add_argument("--mode", type = str, default = "+voc", help = "the folder path of image or \"+voc\" or \"+coco\" or None")
    opt = parser.parse_args()

    Tester(gpu_id = opt.gpu_id, weight_path = opt.weight_path, mode = opt.mode).test()

