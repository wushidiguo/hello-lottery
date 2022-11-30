import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from detector import Detector
from recognizer import Recognizer
from checker import Checker
from utils import *

class Lottery:

    def __init__(
        self,
        detector = "./checkouts/detection.pt",
        recognizer = "./checkouts/recognition.pt",
        detect_conf_thres = 0.25, 
        detect_iou_thres = 0.45, 
        cert_ = "cert_.txt",
        timeout = 5,
        cuda = True
    ):
        self.detector_ = detector
        self.detect_conf_thres = detect_conf_thres
        self.detect_iou_thres = detect_iou_thres
        self.recognizer_ = recognizer
        self.cert_ = cert_
        self.timeout = timeout
        self.device = "cuda" if torch.cuda.is_available() and cuda else "cpu"

        self.init_detector()
        self.init_recognizer()
        self.init_checker()

        self.last_result = None

    def init_detector(self):
        self.detector = Detector(self.detector_, self.detect_conf_thres, self.detect_iou_thres, self.device)

    def init_recognizer(self):
        self.recognizer = Recognizer(self.recognizer_, self.device)

    def init_checker(self):
        self.checker = Checker.from_file(self.cert_, timeout=self.timeout)

    def __call__(self, img):
        p = Path(img)
        if not p.is_file():
            raise FileNotFoundError()
        p = str(p.absolute())
        img0 = cv2.imread(p)  # BGR
        assert img0 is not None, 'Cannot read image ' + str(p)

        detection = self.detector(img0)
        if not detection:
            return

        code, issue_, numbers_ = detection

        issue = crop(img0, issue_)[0]
        numbers_ = sort_box(numbers_)
        numbers_ = crop(img0, numbers_)

        numbers_.append(issue)

        recognition = self.recognizer(numbers_)
        if not recognition:
            return
        
        issue, _ = recognition.pop()
        numbers = recognition

        numbers = [num[0] for num in numbers]

        numbers = number_process(numbers, code)
        
        issue = issue_process(issue)

        hits, winning = self.checker(code, issue, numbers)

        self.last_result = code, issue, winning, numbers, hits

        return self.last_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", type=str, default="./checkouts/detection.pt", help="detector model file")
    parser.add_argument("--recognizer", type=str, default="./checkouts/recognition.pt", help="recognizer model file")
    parser.add_argument("--detect_conf_thres", type=float, default=0.25, help="detection confidence threshold")
    parser.add_argument("--detect_iou_thres", type=float, default=0.45, help="detection iou threshold")
    parser.add_argument("--cert_", type=str, default="./cert_.txt", help="API infomation")
    parser.add_argument("--timeout", type=int, default=5, help="timeout for waiting response.")
    parser.add_argument("--cuda", action="store_true", help="use cuda or cpu")
    parser.add_argument("--image", type=str, help="image with lottery in it")

    opt = parser.parse_args()

    l = Lottery(
        detector=opt.detector,
        recognizer=opt.recognizer,
        detect_conf_thres=opt.detect_conf_thres,
        detect_iou_thres=opt.detect_iou_thres,
        cert_=opt.cert_,
        timeout=opt.timeout,
        cuda=opt.cuda
    )

    assert opt.image, "Please specify an image containing lottery."

    result = l(opt.image)
    if not result:
        print("Sorry, something is wrong. You may try again.")
    else:
        pprint(*result)
        
    



