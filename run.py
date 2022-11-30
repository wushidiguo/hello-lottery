import sys
from threading import Thread

from PyQt5 import QtWidgets, QtGui, QtCore

from lottery import Lottery


class Main(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.engine_status = 2  # 2:未启动，1:启动成功，0:启动失败
        self.initUI()
        self.initEngine()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()
        self.label = ClickableLabel(self)
        self.table = QtWidgets.QTableView(self)
        layout.addWidget(self.label, stretch=1)
        layout.addWidget(self.table, stretch=1)

        self.label.setFrameStyle(1)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setPixmap(QtGui.QPixmap("./icons/add.png").scaled(50, 50))
        self.label.clicked_signal_connect(self.open)

        self.table.horizontalHeader().hide()

        self.status = self.statusBar()

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
        self.setGeometry(100, 100, 400, 600)
        self.setWindowTitle("Hello Lottery!")
        self.setWindowIcon(QtGui.QIcon("./icons/lottery.png"))

    def initEngine(self):
        self.status.showMessage("正在启动识别引擎...")
        engine_thread = EngineThread(self)
        engine_thread.start()
    
    def engine_started(self, flg):
        self.engine_status = flg
        if flg:
            self.status.showMessage(f"识别引擎({self.engine.device})启动成功！")
        else:
            self.status.showMessage("识别引擎启动失败！请退出重试！")
            
    def open(self):

        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File',".","Images (*.png *.xpm *.jpg *.bmp *.gif)")[0]

        if filename:
            if self.engine_status == 2:
                QtWidgets.QMessageBox.information(self, "Info", "请等待识别引擎启动完成！")
                return
            elif self.engine_status == 0:
                QtWidgets.QMessageBox.information(self, "Warning", "识别引擎启动失败！请退出重试！")
                return

            self.display_image(filename)
            self.process(filename)

    def process(self, filename):
        self.status.showMessage("识别中...")
        process_thread = ProcessThread(self.engine, filename)
        process_thread.signal.connect(self.display_results)
        process_thread.start()
        process_thread.exec()   # 使线程进入事件循环状态，否则程序直接崩溃
    
    def display_image(self, filename):
        pixmap = QtGui.QPixmap(filename).scaled(self.label.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
        self.label.setPixmap(pixmap)

    def display_results(self, flg):
        if not flg:
            QtWidgets.QMessageBox.information(self, "Warning", "识别过程出错！请重试！")
            self.status.clearMessage()
            return
        if not self.engine.last_result:
            QtWidgets.QMessageBox.information(self, "Warning", "没有识别到彩票信息！请调整图片后重试！")
            self.status.clearMessage()
            return
        self.status.showMessage("识别完成，请核对结果！")

        code, issue, winning, numbers, hits = self.engine.last_result
        self.engine.last_result = None

        game_type = numbers["game_type"]
        convert = {
        "single" : "单式",
        "compound" : "复式",
        "complex" : "胆拖"
        }
        header_labels = ["彩票类型", "开奖期", "开奖号码", "玩法"]
        model = QtGui.QStandardItemModel()
        model.setVerticalHeaderLabels(header_labels)
        model.setItem(0, 0, QtGui.QStandardItem("双色球" if code == "ssq" else "超级大乐透"))
        model.setItem(1, 0, QtGui.QStandardItem(issue))
        model.setItem(2, 0, QtGui.QStandardItem(" ".join(winning[0] + ["+"] + winning[1])))
        model.setItem(3, 0, QtGui.QStandardItem(convert[game_type]))
        i = 4
        numbers = numbers["numbers"]
        if game_type in ["single", "compound"]:
            series = "①②③④⑤⑥⑦⑧⑨⑩"
            for s, num, hit in zip(series, numbers, hits):
                model.setVerticalHeaderItem(i, QtGui.QStandardItem(s))
                model.setItem(i, 0, QtGui.QStandardItem(" ".join(num[0] + [f"(中{len(hit[0])})"] + ["+"] + num[1] + [f"(中{len(hit[1])})"])))
                i += 1
        else:
            numbers = numbers[0]
            hits = hits[0]
            if code == "cjdlt":
                series = ["前区胆", "前区拖", "后区胆", "后区拖"]
                for s, num, hit in zip(series, numbers, hits):
                    model.setVerticalHeaderItem(i, QtGui.QStandardItem(s))
                    model.setItem(i, 0, QtGui.QStandardItem(" ".join(num) + f" (中{len(hit)})"))
                    i += 1
            else:
                series = ["红胆", "红拖", "蓝单" if len(numbers[3]) == 1 else "蓝复"]
                for s, num, hit in zip(series, [numbers[i] for i in [0, 1, 3]], [hits[i] for i in [0, 1, 3]]):
                    model.setVerticalHeaderItem(i, QtGui.QStandardItem(s))
                    model.setItem(i, 0, QtGui.QStandardItem(" ".join(num) + f" (中{len(hit)})"))
                    i += 1
        self.table.setModel(model)
        self.table.resizeColumnToContents(0)

        
class ClickableLabel(QtWidgets.QLabel):

    label_clicked_signal = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

    def mouseReleaseEvent(self, QMouseEvent):
        self.label_clicked_signal.emit()
        
    def clicked_signal_connect(self, func):
        self.label_clicked_signal.connect(func)


class EngineThread(Thread):
    def __init__(self, obj):
        super().__init__()
        self.obj = obj

    def run(self):
        try:
            engine = Lottery()
        except Exception as e:
            self.obj.engine = None
            self.obj.engine_started(0)
            print(e)
        else:
            self.obj.engine = engine
            self.obj.engine_started(1)


class ProcessThread(QtCore.QThread):
    signal = QtCore.pyqtSignal(int)
    def __init__(self, engine, filename):
        super().__init__()
        self.engine = engine
        self.filename = filename

    def run(self):
        try:
            self.engine(self.filename)
        except Exception as e:
            self.signal.emit(0)
            print(e)
        else:
            self.signal.emit(1)


def main():
    app = QtWidgets.QApplication(sys.argv)

    main = Main()
    main.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()