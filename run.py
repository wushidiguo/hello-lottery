import sys
from threading import Thread

from PyQt5 import QtWidgets, QtGui, QtCore

from lottery import Lottery
from utils import *


class Main(QtWidgets.QMainWindow):
    
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.engine_status = 2  # 2:未启动，1:启动成功，0:启动失败
        self.initUI()
        self.initEngine()

    def initUI(self):
        layout = QtWidgets.QGridLayout()
        self.label = ClickableLabel(self)
        self.table = QtWidgets.QTableView(self)
        self.tips = QtWidgets.QLabel(self)
        self.confirm_button = QtWidgets.QPushButton(self)
        layout.addWidget(self.label, 0, 0, 5, 5)
        layout.addWidget(self.table, 5, 0, 5, 4)
        layout.addWidget(self.tips, 5, 4, 4, 1)
        layout.addWidget(self.confirm_button, 9, 4, 1, 1)

        self.label.setFrameStyle(1)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setPixmap(QtGui.QPixmap("./icons/add.png").scaled(50, 50))
        self.label.clicked_signal_connect(self.open)

        self.table.horizontalHeader().hide()

        self.tips.setText("Tips: \n请核对识别\n结果，如有\n错误，请在\n表格中修改。\n确认无误后，\n点击下方按\n钮查询中奖\n情况。👇")

        self.confirm_button.setText("信息无误\n中奖查询")
        self.confirm_button.clicked.connect(self.confirmed)

        self.statusBar = self.statusBar()

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
        self.setGeometry(100, 100, 500, 800)
        self.setWindowTitle("Hello Lottery!")
        self.setWindowIcon(QtGui.QIcon("./icons/lottery.png"))

    def initEngine(self):
        self.statusBar.showMessage("正在启动识别引擎...")
        engine_thread = EngineThread(self)
        engine_thread.start()
    
    def engine_started(self, flg):
        self.engine_status = flg
        if flg:
            self.statusBar.showMessage(f"识别引擎({self.engine.device})启动成功！")
        else:
            self.statusBar.showMessage("识别引擎启动失败！请退出重试！")
            
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
        self.statusBar.showMessage("识别中...")
        self.confirm_button.setEnabled(False)
        process_thread = ProcessThread(self.engine, filename)
        process_thread.signal.connect(self.display_results)
        process_thread.start()
        process_thread.exec()   # 使线程进入事件循环状态，否则程序直接崩溃
    
    def display_image(self, filename):
        pixmap = QtGui.QPixmap(filename)
        if pixmap.width() > pixmap.height():
            pixmap = QtGui.QPixmap(filename).scaledToWidth(self.label.width(), QtCore.Qt.SmoothTransformation)
        else:
            pixmap = QtGui.QPixmap(filename).scaledToHeight(self.label.height(), QtCore.Qt.SmoothTransformation)
        self.label.setPixmap(pixmap)

    def display_results(self, signal):
        self.confirm_button.setEnabled(True)
        if isinstance(signal, Exception):
            QtWidgets.QMessageBox.information(self, "Warning", str(signal))
            self.statusBar.clearMessage()
            return

        if len(signal) == 3:
            self.statusBar.showMessage("识别完成，请核对结果！")
        
        else:
            self.statusBar.showMessage("查询完成！")

        data = Result.fromTuple(signal)
        model = TableModel(data, self)
        self.table.setModel(model)
        self.table.resizeColumnToContents(0)

    def confirmed(self):
        if not isinstance(self.table.model(), TableModel):
            return
        confirmed_results = self.table.model().results.toTuple()
        self.check(confirmed_results)
    
    def check(self, query):
        self.statusBar.showMessage("查询中...")
        self.confirm_button.setEnabled(False)
        process_thread = CheckThread(self.engine, query)
        process_thread.signal.connect(self.display_results)
        process_thread.start()
        process_thread.exec() 

        
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
    signal = QtCore.pyqtSignal(object)
    def __init__(self, engine, filename):
        super().__init__()
        self.engine = engine
        self.filename = filename

    def run(self):
        try:
            results = self.engine(self.filename, recognition_only=True)
            if not results:
                raise(MissingInfoException("没有识别到彩票信息！请调整图片后重试！"))
        except Exception as e:
            self.signal.emit(e)
        else:
            self.signal.emit(results)

class CheckThread(QtCore.QThread):
    signal = QtCore.pyqtSignal(object)
    def __init__(self, engine, query: tuple):
        super().__init__()
        self.engine = engine
        self.query = query

    def run(self):
        try:
            results = self.engine.check(*self.query)
        except Exception as e:
            self.signal.emit(e)
        else:
            self.signal.emit(results)           


def main():
    app = QtWidgets.QApplication(sys.argv)

    main = Main()
    main.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()