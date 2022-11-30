import numpy as np
import torch

from models.yolo.utils.datasets import letterbox
from models.yolo.utils.general import check_img_size, non_max_suppression, scale_coords
from models.yolo.utils.torch_utils import select_device
from models.yolo.model import Model
from models.yolo.experimental import attempt_load


class Detector:

    def __init__(
        self,
        model_file,
        conf_thres = 0.25, 
        iou_thres = 0.45, 
        device = "cpu"
    ):
        self.device = select_device(device)
        weights = torch.load(model_file, self.device)
        self.model = weights["model"]
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(640, s=self.stride)

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        self.half = device != "cpu"
        if self.half:
            self.model.half()
        self.model.eval()

    def __call__(
        self, 
        img
    ):
        shape = img.shape
        img = letterbox(img, self.imgsz, stride=self.stride)[0]   # resize & padding
        img = img[:, :, ::-1].transpose(2, 0, 1)    # hwc(bgr) -> hwc(rgb) -> c(rgb)hw
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0) 

        with torch.no_grad():
            pred = self.model(img, augment=True)[0]

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)[0]

        if not len(pred):
            return
        
        # Scale boxes size back
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], shape).round()

        pred_ = pred[:, [0, 1, 2, 3, 5]].to("cpu", int).numpy()
        cls = np.unique(pred_[:, -1], return_index=False)

        if 0 in cls:
            code = "ssq"
        elif 2 in cls:
            code = "cjdlt"
        else:
            return
        
        if not 1 in cls and 3 in cls:
            return

        numbers = pred_[pred_[:, -1] == 1]
        
        issue = pred_[pred_[:, -1] == 3]
        
        return code, issue, numbers
        


        