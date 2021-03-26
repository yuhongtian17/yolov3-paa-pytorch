from timeit import default_timer as timer

import cv2
import argparse

import config.yolov3_config as cfg
from eval.evaluator import Evaluator
from model.yolov3 import Model_yolov3
from utils.gpu import *
from utils.tools import *
from utils.visualize import *


class Videoer(object):
    """
    """
    def __init__(self, gpu_id = 0, weight_path = None, video_path = None, output_dir = None, mode = None):
        self.gpu_id = gpu_id
        self.weight_path = weight_path
        self.video_path = video_path
        self.output_dir = output_dir
        self.mode = mode

        if self.mode == "+voc":
            self.__classes = cfg.VOC_CLASS["CLASSES"]
            self.__num_classes = cfg.VOC_CLASS["NUM_CLASSES"]
        else:
            self.__classes = cfg.COCO_CLASS["CLASSES"]
            self.__num_classes = cfg.COCO_CLASS["NUM_CLASSES"]

        self.__device = select_device(self.gpu_id) # 先确定gpu编号
        self.__model = Model_yolov3(num_classes = self.__num_classes).to(self.__device) # 再加载初始模型，并送到设备上
        self.__load_model_weights(self.weight_path) # 再加载训练好的模型


    def __load_model_weights(self, weight_path):
        print("loading weight file from: {}".format(weight_path))

        chkpt = torch.load(weight_path, map_location = self.__device)
        self.__model.load_state_dict(chkpt)
        del chkpt

        print("loading weight file is done")


    def video_detection(self):

        vid = cv2.VideoCapture(self.video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (
            int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        isOutput = True if self.output_dir != "" else False
        if isOutput:
            print(
                "!!! TYPE:",
                type(self.output_dir),
                type(video_FourCC),
                type(video_fps),
                type(video_size),
            )
            out = cv2.VideoWriter(
                self.output_dir, video_FourCC, video_fps, video_size
            )
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        while True:
            return_value, frame = vid.read()

            evaluator_imgs = Evaluator(self.__model, self.__classes, self.__num_classes, visual = False)
            bboxes_prd = evaluator_imgs.get_bbox(frame)
            if bboxes_prd.shape[0] != 0:
                boxes = bboxes_prd[..., :4]
                class_inds = bboxes_prd[..., 5].astype(np.int32)
                scores = bboxes_prd[..., 4]

                visualize_boxes(image = frame, boxes = boxes, labels = class_inds, probs = scores, class_labels = self.__classes)

            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0

            cv2.putText(frame, text = fps, org = (3, 15), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.50, color = (255, 0, 0), thickness = 2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", frame)

            if isOutput:
                out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type = int, default = 0, help = "gpu id")
    parser.add_argument("--weight_path", type = str, default = "./weight/best.pth", help = "the file path of weight or None")
    parser.add_argument("--video_path", type = str, default = "./aaaa.avi", help = "the file path of weight or None")
    parser.add_argument("--output_dir", type = str, default = "./data/", help = "the file path of weight or None")
    parser.add_argument("--mode", type = str, default = "+voc", help = "the folder path of image or \"+voc\" or \"+coco\" or None")
    opt = parser.parse_args()

    Videoer(gpu_id = opt.gpu_id, weight_path = opt.weight_path, video_path = opt.video_path, output_dir = opt.output_dir, mode = opt.mode).video_detection()

