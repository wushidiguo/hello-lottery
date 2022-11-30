from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

from utils import *

class Recognizer:

    def __init__(
        self,
        model_file,
        device = "cpu",
        ):
        weights = torch.load(model_file)
        self.model = weights["model"]
        self.converter = weights["converter"]
        self.opt = self.model.opt
        self.imgH = self.opt.imgH
        self.imgW = self.opt.imgW
        self.input_channel = self.opt.input_channel
        self.device = device

        _ = self.model.to(device)
        self.model.eval()

    def __call__(
        self,
        imgs
        ):
        results = []
        transform = NormalizePAD((self.input_channel, self.imgH, self.imgW))
        with torch.no_grad():
            for img in imgs:
                img = Image.fromarray(img).convert("L")
                w, h = img.size
               
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                img = img.resize((resized_w, self.imgH), Image.BICUBIC)
                img = transform(img)
                img = img.unsqueeze(0)
                img = img.to(self.device)

                text_for_pred = torch.LongTensor(1, w // 10 + 1).fill_(0).to(self.device)

                preds = self.model(img, text_for_pred)

                preds_size = [preds.size(1)]

                preds_prob = softmax(preds, dim=-1).squeeze().cpu().detach().numpy()

                values = preds_prob.max(axis=-1)
                indices = preds_prob.argmax(axis=-1)

                preds_str = self.converter.decode_greedy(indices.ravel(), preds_size)[0]
                confidence_score = custom_mean(values)

                results.append([preds_str, confidence_score])

        return results




