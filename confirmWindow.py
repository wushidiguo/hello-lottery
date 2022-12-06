from PyQt5 import QtWidgets, QtGui, QtCore

from utils import *


class ConfirmWindow(QtWidgets.QDialog):
    def __init__(self, pixmap, results, parent=None):
        super().__init__(parent)
        self.pixmap = pixmap
        self.results = results
        self.parent = parent
        self.initUI()
        self.display_results()

    def initUI(self):
        layout = QtWidgets.QGridLayout()
        self.label = QtWidgets.QLabel(self)
        self.table = QtWidgets.QTableView(self)
        self.ok = QtWidgets.QPushButton("结果确认完毕，下一步→")
        self.cancel = QtWidgets.QPushButton("取消并返回")
        
        layout.addWidget(self.label, stretch=1)
        layout.addWidget(self.table, stretch=1)

        self.label.setFrameStyle(1)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setPixmap(self.pixmap)

        self.table.horizontalHeader().hide()
        
        self.setWindowTitle("请核对识别信息，如有错误请手动修改")
        self.setGeometry(100, 100, 400, 600)
        self.setLayout(layout)
        
    
    def display_results(self):
        self.results

