# hello-lottery
通过AI快速识别彩票信息，并给出中奖结果。目前支持体彩超级大乐透和福彩双色球两种彩票，支持单式、复式、胆拖玩法。

## 运行环境
测试环境为Windows 11和Ubuntu 20.04，python版本为3.8.10，默认使用GPU加速，纯CPU环境下亦可运行。使用时需要联网获取开奖数据。

## 使用步骤
```bash
git clone https://github.com/wushidiguo/hello-lottery

cd hello-lottery

pip install -r requirements.txt
```
**通过图形界面使用**
```bash
python run.py
```
点击按钮选择对应彩票图片即可。

**通过命令行使用**
```bash
python lottery.py --image ##.jpg
```

**脚本内使用**
```python
from lottery import Lottery

l = Lottery()
results = l("##.jpg")
```

## 说明
彩票识别分为两步：
1. 关键信息检测和定位，使用了[yolov7](https://github.com/WongKinYiu/yolov7)的标准模型，训练过程中使用了收集于网上的约200张彩票图像（大乐透和双色球各100张），另外使用了50张图像作为验证集；
2. 文字ocr，使用了[easyocr](https://github.com/JaidedAI/EasyOCR)提供的模型架构，即VGG+BiLSTM+CTC，使用了约1000张彩票截图数据作为训练集，约250张截图数据作为验证集，同时在每个epoch，加入与真实数据同等数量的人造数据，一同进行训练；

此外，使用了https://www.mxnzp.com/ 提供的免费接口获取彩票开奖信息。




