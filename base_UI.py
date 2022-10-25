from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui, QtWidgets


class base_window(QWidget):
    def __init__(self):
        super(base_window, self).__init__()
        self.font = QtGui.QFont()
        self.font.setFamily("標楷體")
        self._translate = QtCore.QCoreApplication.translate

    def show_window(self, other_window):
        self.show()
        other_window.hide()


class QTComboBox:
    def __init__(self, name, window, item_list):
        self.window = window
        self.font = QtGui.QFont()
        self.font.setFamily("標楷體")
        self.item_list = item_list
        self.ComboBox = QComboBox()
        self.ComboBox.setFont(self.font)
        self.ComboBox.setStyleSheet(
            "QComboBox{border:2px groove gray;border-radius:10px;padding:2px 4px;background-color:#F8F8FF;font-size:18px;}"
            "QComboBox:hover{background-color:rgb(240, 248, 255);border-color:rgb(100, 149, 237);font-size:18px;}"
        )
        self.ComboBox.setObjectName(name)
        self.ComboBox.addItems(self.item_list)
        self.ComboBox.setContentsMargins(0, 0, 0, 0)


class QTLabel:
    def __init__(self, name, window, text):
        self.window = window
        self.font = QtGui.QFont()
        self.font.setFamily("標楷體")
        self.label = QtWidgets.QLabel()
        self.label.setFont(self.font)
        self.label.setLayoutDirection(Qt.LeftToRight)
        self.label.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter)
        self.label.setStyleSheet("QLabel{font-size:18px;}")
        self.label.setObjectName(name)
        self.label.setText(QCoreApplication.translate(self.window, text))

    def set_text(self, text):
        self.label.setText(QCoreApplication.translate(self.window, text))


class QTLineEdit:
    def __init__(self, name, window):
        self.lineEdit = QtWidgets.QLineEdit()
        self.window = window
        self.lineEdit.setObjectName(name)
        self.lineEdit.setStyleSheet("#%s{border:2px groove gray;border-radius:10px;padding:2px 4px;font-size:18px;}" % name)

    def set_text(self, text):
        self.lineEdit.setText(QCoreApplication.translate(self.window, text))


class QTButton:
    def __init__(self, name, window):
        self.font = QtGui.QFont()
        self.font.setFamily("標楷體")
        self.window = window
        self.pushButton = QtWidgets.QPushButton()
        self.pushButton.setFont(self.font)
        self.pushButton.setObjectName(name)
        self.pushButton.setStyleSheet(
            "QPushButton{border:2px groove gray;border-radius:10px;padding:2px 4px;background-color:#F8F8FF;font-size:18px;}"
            "QPushButton:hover{background-color:rgb(240, 248, 255);border-color:rgb(100, 149, 237);font-size:18px;}"
        )

    def set_text(self, text):
        self.pushButton.setText(QCoreApplication.translate(self.window, text))