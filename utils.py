import math
import re
import logging
from dataclasses import dataclass

from natsort import natsort_key
import cv2
import numpy as np
import torch
from torchvision import transforms
from PyQt5 import QtCore
from PyQt5.QtCore import Qt

log = logging.getLogger(__name__)


def imread(
    filename, 
    flags=cv2.IMREAD_COLOR, 
    dtype=np.uint8
):
    '''
    读取图像文件，解决cv2.imread()对非英文命名文件报错问题。
    '''
    try: 
        img = np.fromfile(filename, dtype)
        img = cv2.imdecode(img, flags)
        return img
    except Exception as e:
        print(e)
        return

def sort_box(
    boxes
):
    '''
    对box按照在图片上的位置从上到下进行排序。
    '''
    return boxes[boxes[:, 1].argsort(), :]

def crop(
    img,
    boxes
):
    '''
    按给定的box对img进行裁切，并返回相应的子图list。
    '''
    if boxes.ndim == 1:
        boxes = [boxes]
    results = []
    for xyxy in boxes:
        results.append(img[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2], :])
    return results

class NormalizePAD:
    '''
    对进入recognizer进行ocr识别的图片进行padding等预处理。
    '''
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img

def custom_mean(x):
    '''
    计算ocr的平均confidence score。
    '''
    return x.prod()**(2.0/np.sqrt(len(x)))

class AttrDict(dict):
    '''
    保存模型参数。
    '''
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def number_process(numbers, code):
    '''
    彩票号码前处理。
    '''
    assert code in ["ssq", "cjdlt"], f"Code {code} is illegal."

    first_line = numbers[0].strip()

    pattern1 = re.compile("\d+")    # 匹配数字
    pattern2 = re.compile("\*\d+$|\(.*\)|") # 匹配倍数，用于从号码中删除倍数

    def match_fill(string):
        matched = pattern1.findall(string)
        return [num.zfill(2) for num in matched]
    
    results = []
    if first_line.startswith(("A", "1)")):
        game_type = "single"
        for line in numbers:
            matched = match_fill(line[1:])
            if len(matched) < 7:
                raise MissingInfoException("Matched numbers not enough.")
            if code == "ssq":
                red = matched[: 6]
                blue = [matched[6]]
            else:
                red = matched[: 5]
                blue = matched[5 : 7]
            results.append((red, blue))

    elif first_line.startswith(("红胆", "前区胆")):
        game_type = "complex"
        red_required = []
        red_optional = []
        blue_required = []
        blue_optional = []
        for line in numbers:
            line = line.strip()
            if line.startswith(("红胆", "前区胆")):
                red_required = match_fill(line)
                previous = red_required
            elif line.startswith(("红拖", "前区拖")):
                red_optional = match_fill(line)
                previous = red_optional
            elif line.startswith(("后区胆")):
                blue_required = match_fill(line)
                previous = blue_required
            elif line.startswith(("蓝单", "后区拖", "蓝复")):
                blue_optional = match_fill(line)
                previous = blue_optional
            elif line.startswith("倍数"):
                continue
            else:
                previous.extend(match_fill(line))
        if len(red_required) + len(red_optional) + len(blue_required) + len(blue_optional) <= 7:
            raise MissingInfoException("Matched numbers not enough.")
        results.append((red_required, red_optional, blue_required, blue_optional))

    elif first_line.startswith(("前区", "红区", "红单", "红复")):
        game_type = "compound"
        red = []
        blue = []
        for line in numbers:
            line = line.strip()
            if line.startswith(("前区", "红区", "红单", "红复")):
                red = match_fill(line)
                previous = red
            elif line.startswith(("后区", "蓝区", "蓝单", "蓝复")):
                blue = match_fill(line)
                previous = blue
            elif line.startswith("倍数"):
                continue
            else:
                previous.extend(match_fill(line))
        if len(red) + len(blue) <= 7:
            raise MissingInfoException("Matched numbers not enough.")
        results.append((red, blue))

    else:
        line = "".join(numbers)
        line = pattern2.sub("", line)
        matched = match_fill(line)
        if len(matched) < 7:
            raise MissingInfoException("Matched numbers not enough.")
        elif len(matched) == 7:
            game_type = "single"
            if code == "ssq":
                red = matched[: 6]
                blue = [matched[6]]
            else:
                red = matched[: 5]
                blue = matched[5 : 7]
            results.append((red, blue))
        else:
            game_type = "compound"
            section_con = None
            cons = ["-", "+", "*"]
            for con in cons:
                if con in line and line.count(con) == 1:
                    section_con = con
                    break
            if not section_con:
                raise MissingInfoException("Sections connector not found.")
            red_half, blue_half = line.split(section_con)
            red = match_fill(red_half)
            blue = match_fill(blue_half)
            results.append((red, blue))

    return {
    "code" : code, 
    "game_type" : game_type, 
    "numbers" : results
    }

class MissingInfoException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


def issue_process(issue_string):
    '''
    开奖/销售期处理。
    '''
    issue = re.findall("\d+", issue_string)
    if len(issue) > 1:
        return
    return issue[0]

def winning_process(winning_number, code):
    '''
    中奖号码处理。
    '''
    pattern = re.compile("\d+")
    matched = pattern.findall(winning_number)
    if len(matched) < 7:
        raise MissingInfoException("Matched numbers not enough.")
    if code == "ssq":
        red = matched[: 6]
        blue = [matched[6]]
    else:
        red = matched[: 5]
        blue = matched[5 : 7]
    return red, blue

def hit_check(numbers, winning_numbers):
    '''
    中奖号码匹配。
    '''
    log.info("Winning numbers are: ", winning_numbers)
    red_win, blue_win = winning_numbers
    hits = []
    if numbers["game_type"] == "single" or numbers["game_type"] == "compound":
        for number in numbers["numbers"]:
            log.info("User numbers are: ", number)
            red, blue = number
            red_hit = sorted(list(set(red) & set(red_win)), key=natsort_key)
            blue_hit = sorted(list(set(blue) & set(blue_win)), key=natsort_key)
            log.info("Hit numbers are: ", (red_hit, blue_hit))
            hits.append((red_hit, blue_hit))
    else:
        for number in numbers["numbers"]:
            log.info("User numbers are: ", number)
            red_required, red_optional, blue_required, blue_optional = number
            red_required_hit = sorted(list(set(red_required) & set(red_win)), key=natsort_key)
            red_optional_hit = sorted(list(set(red_optional) & set(red_win)), key=natsort_key)
            blue_required_hit = sorted(list(set(blue_required) & set(blue_win)), key=natsort_key)
            blue_optional_hit = sorted(list(set(blue_optional) & set(blue_win)), key=natsort_key)
            log.info("Hit numbers are: ", (red_required_hit, red_optional_hit, blue_required_hit, blue_optional_hit))
            hits.append((red_required_hit, red_optional_hit, blue_required_hit, blue_optional_hit))
    return hits


def pprint(code, issue, winning, numbers, hits):
    '''
    增加结果打印的可读性。
    '''
    print("双色球" if code == "ssq" else "超级大乐透")
    print("开奖期:", issue)

    convert = {
        "single" : "单式",
        "compound" : "复式",
        "complex" : "胆拖"
        }

    if code == "ssq":
        print(f"开奖号码: (红){' '.join(winning[0])} (蓝){' '.join(winning[1])}")

        game_type = numbers["game_type"]
        print("玩法:", convert[game_type])

        print("识别号码")
        if game_type in ["single", "compound"]:
            for num in numbers["numbers"]:
                print(f"(红){' '.join(num[0])} (蓝){' '.join(num[1])}")
        else:
            for num in numbers["numbers"]:
                print(f"(红胆){' '.join(num[0])}")
                print(f"(红拖){' '.join(num[1])}")
                print(f"({'蓝单' if len(num[3] == 1) else '蓝复'}){' '.join(num[3])}")
        print("命中数")
        if game_type in ["single", "compound"]:
            for hit in hits:
                print(f"(红)中{len(hit[0])} (蓝)中{len(hit[1])}")
        else:
            for hit in hits:
                print(f"(红胆)中{len(hit[0])}")
                print(f"(红拖)中{len(hit[1])}")
                print(f"(蓝)中{len(hit[3])}")
    else:
        print(f"开奖号码: (前区){' '.join(winning[0])} (后区){' '.join(winning[1])}")

        game_type = numbers["game_type"]
        print("玩法:", convert[game_type])

        print("识别号码")
        if game_type in ["single", "compound"]:
            for num in numbers["numbers"]:
                print(f"(前区){' '.join(num[0])} (后区){' '.join(num[1])}")
        else:
            for num in numbers["numbers"]:
                print(f"(前区胆){' '.join(num[0])}")
                print(f"(前区拖){' '.join(num[1])}")
                print(f"(后区胆){' '.join(num[2])}")
                print(f"(后区拖){' '.join(num[3])}")
        print("命中数")
        if game_type in ["single", "compound"]:
            for hit in hits:
                print(f"(前区)中{len(hit[0])} (后区)中{len(hit[1])}")
        else:
            for hit in hits:
                print(f"(前区胆)中{len(hit[0])}")
                print(f"(前区拖)中{len(hit[1])}")
                print(f"(后区胆)中{len(hit[2])}")
                print(f"(后区拖)中{len(hit[3])}")


class Result:
    '''
    要允许用户修改识别结果，就要有一个对应的数据结构作为“后台数据”和“前台表格”的桥梁。
    因为要达到的效果是不同彩票、不同玩法显示结果的格式不同，
    导致人为修改数据时的处理很不简洁，但暂时没有想到更好的方法。
    '''
    def __init__(self, code: str, issue: str, game_type: str, numbers: list, winning: tuple = None, hits: list = None):
        self.code = code
        self.issue = issue
        self.game_type = game_type
        self.numbers = numbers
        self.winning = winning
        self.hits = hits
        self.fixed_headers = ["彩票类型", "开奖期", "开奖号码", "玩法"]
        self.fixed_row = len(self.fixed_headers)

    @classmethod
    def fromTuple(self, t):
        if len(t) == 3:
            code, issue, numbers_ = t
            game_type = numbers_["game_type"]
            numbers = numbers_["numbers"]
            return Result(code, issue, game_type, numbers)
        else:
            code, issue, winning, numbers_, hits = t
            game_type = numbers_["game_type"]
            numbers = numbers_["numbers"]
            return Result(code, issue, game_type, numbers, winning, hits)

    def toTuple(self):
        return self.code, self.issue, {"code": self.code, "game_type": self.game_type, "numbers": self.numbers}

    def codeConvert(self, code):
        return "双色球" if code == "ssq" else "超级大乐透"

    def codeRevert(self, s):
        return "ssq" if s == "双色球" else "cjdlt"
    
    def gameConvert(self, game):
        convert = {
        "single": "单式",
        "compound": "复式",
        "complex": "胆拖"
        }
        return convert[game]

    def gameRevert(self, s):
        revert = {
            "单式": "single",
            "复式": "compound",
            "胆拖": "complex"
        }
        return revert[s]

    def numbersConvert(self):
        if self.game_type in ["single", "compound"]:
            return [" ".join(num[0] + ["+"] + num[1]) for num in self.numbers]
        else:
            if self.code == "cjdlt":
                return [" ".join(num) for num in self.numbers[0]]
            else:
                return [" ".join(num) for num in [self.numbers[0][i] for i in [0, 1, 3]]]
    
    def hitsConvert(self):
        if self.game_type in ["single", "compound"]:
            return ["中" + str(len(hit[0])) + " + " + str(len(hit[1])) for hit in self.hits]
        else:
            if self.code == "cjdlt":
                return ["中" + str(len(hit)) for hit in self.hits[0]]
            else:
                return ["中" + str(len(hit)) for hit in [self.hits[0][i] for i in [0, 1, 3]]]

    def toHeaderList(self):
        if self.game_type in ["single", "compound"]:
            return self.fixed_headers + list("①②③④⑤⑥⑦⑧⑨⑩")[: len(self.numbers)]
        else:
            if self.code == "ssq":
                return self.fixed_headers + ["红胆", "红拖", "蓝单" if len(self.numbers[0][3]) == 1 else "蓝复"]
            else:
                return self.fixed_headers + ["前区胆", "前区拖", "后区胆", "后区拖"]
    
    def getData(self, index):
        row, col = index.row(), index.column()
        if col == 0:
            if row == 0:
                return self.codeConvert(self.code)
            if row == 1:
                return self.issue
            if row == 2:
                return " ".join(self.winning[0] + ["+"] + self.winning[1]) if self.winning else "点击查询按钮自动获取"
            if row == 3:
                return self.gameConvert(self.game_type)
            return self.numbersConvert()[row - self.fixed_row]
        elif col == 1 and self.hits and row >= self.fixed_row:
            return self.hitsConvert()[row - self.fixed_row]

    def setData(self, index, text):
        row, col = index.row(), index.column()
        if col != 0:
            return False    # 第一列以外不能修改

        if row >= len(self.toHeaderList()):
            return False

        text = text.strip()

        if row == 0:
            if text in ["超级大乐透", "双色球"]:
                self.code = self.codeRevert(text)
                return True
            return False

        if row == 1:
            if text.isnumeric():
                self.issue = text
                return True
            return False
        
        if row == 2:
            return False    # 中奖号码不允许修改

        if row == 3:
            return False    # 玩法不允许修改

        if self.game_type in ["single", "compound"]:
            splits = text.split("+")
            if len(splits) != 2:
                return False
            s1, s2 = splits
            s1_, s2_ = s1.split(), s2.split()
            for s in s1_ + s2_:
                if not s.isnumeric():
                    return False
            self.numbers[row - self.fixed_row] = (s1_, s2_)
            return True
        else:
            splits = text.strip().split()
            for s in splits:
                if not s.isnumeric():
                    return False
            if self.code == "ssq" and row == 5:
                target = self.numbers[0][3] # 双色球比大乐透少了一个后区胆
            else:
                target = self.numbers[0][row - self.fixed_row]
            target.clear()  # 号码保存在tuple中，不能直接修改，tuple中的元素是list，可以进行原位修改
            target.extend(splits)
            return True
      

class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.results = results
    
    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            return self.results.getData(index)
    
    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.results.toHeaderList())

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 2
    
    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Vertical:
                return self.results.toHeaderList()[section]
            return "" 

    def setData(self, index, value, role):
        if index.isValid() and role == Qt.ItemDataRole.EditRole:
            return self.results.setData(index, value)
        return False

    def flags(self, index):
        if index.isValid():
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
        return Qt.ItemFlag.NoItemFlags 

   