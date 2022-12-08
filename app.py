from lottery import *
import gradio as gr


l = Lottery()
def demo(img):
    res = ""
    try:
        for i in range(4):
            res = l(img)
            if not res:
                img = np.rot90(img, k=3)
            else:
                break
        if not res:
            raise MissingInfoException("没有检测到彩票信息，请调整图片后重试。")
        res =  str(Result.fromTuple(res))
    except Exception as e:
        res = str(e)
    finally:
        return res
 
iface = gr.Interface(
    fn=demo, 
    inputs="image", 
    outputs="text",
    title="Hello Lottery",
    description="彩票OCR项目，通过神经网络识别彩票信息，给出中奖结果。目前支持体彩超级大乐透和福彩双色球，支持单式、复式、胆拖玩法。",
    allow_flagging="never",
    server_name="0.0.0.0"
    )

iface.launch()