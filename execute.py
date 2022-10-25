import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from control import Controller
from Male_window import Male_window
from Femal_window import Female_window


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.resize(400, 300)
        # main window initialize
        self.setWindowTitle(" 服飾分群 ")
        # set icon
        self.icon = QtGui.QIcon()
        self.icon.addPixmap(QtGui.QPixmap("icon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(self.icon)

        controller = Controller(self)

        pushButton_SS = "QPushButton{border:2px groove gray;border-radius:10px;padding:2px 4px;background-color:#F8F8FF;font-size:18px;}" \
                        "QPushButton:hover{background-color:rgb(240, 248, 255);border-color:rgb(100, 149, 237);font-size:18px;}"

        font = QtGui.QFont()
        font.setFamily("標楷體")

        self.Female_window = Female_window(self)
        self.Male_window = Male_window(self)
        _translate = QtCore.QCoreApplication.translate

        self.male_btn = QPushButton(self)
        self.male_btn.setGeometry(QtCore.QRect(10, 75, 180, 150))
        self.male_btn.setFont(font)
        self.male_btn.setObjectName("female_btn")
        self.male_btn.setStyleSheet(pushButton_SS)
        self.male_btn.setText(_translate("MainWindow", "男裝"))

        self.female_btn = QPushButton(self)
        self.female_btn.setGeometry(QtCore.QRect(210, 75, 180, 150))
        self.female_btn.setFont(font)
        self.female_btn.setObjectName("female_btn")
        self.female_btn.setStyleSheet(pushButton_SS)
        self.female_btn.setText(_translate("MainWindow", "女裝"))
        self.set_btn_connect()

    def set_btn_connect(self):
        self.male_btn.clicked.connect(lambda: self.Male_window.show_window(self))
        self.female_btn.clicked.connect(lambda: self.Female_window.show_window(self))

    def restart(self):
        self.Female_window.hide()
        self.Male_window.hide()

        del self.Female_window
        del self.Male_window
        self.Female_window = Female_window(self)
        self.Male_window = Male_window(self)
        self.set_btn_connect()
        self.show()


if __name__ == '__main__':
     app = QApplication([])
     window = MainWindow()
     window.show()
     sys.exit(app.exec_())
