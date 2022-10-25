from base_UI import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtGui, QtWidgets
from control import Controller


class Male_window(base_window):
    def __init__(self, main_window):
        super(Male_window, self).__init__()
        self.controller = Controller(self)
        self.setWindowTitle(self._translate("Male_window", "Male_window"))
        self.resize(400, 400)
        self.main_window = main_window
        self.seasons_list = ['春夏', '秋冬']
        self.features_list = ['Texture', 'Material', 'Color']

        self.finish = False

        self.season_label = QTLabel("season_label", "Male_window", ">> 選擇季節:")
        self.seasons = QComboBox(self)
        self.seasons.setFont(self.font)
        self.seasons.setStyleSheet(
            "QComboBox{border:2px groove gray;border-radius:10px;padding:2px 4px;background-color:#F8F8FF;font-size:18px;}"
            "QComboBox:hover{background-color:rgb(240, 248, 255);border-color:rgb(100, 149, 237);font-size:18px;}"
        )
        self.seasons.addItems(self.seasons_list)
        self.seasons.setContentsMargins(0, 0, 0, 0)
        grid_layout = QGridLayout()
        h_layout = QHBoxLayout()

        self.feature_label = QTLabel("feature_label", "Male_window", ">> 選擇特徵:")

        self.checkboxes = []
        for i, feature in enumerate(self.features_list):
            self.checkboxes.append(QCheckBox(feature, self))
            self.checkboxes[i].toggle()
            self.checkboxes[i].setStyleSheet(
                "QCheckBox{border:1px groove gray;border-radius:10px;padding:2px 4px;background-color:#F8F8FF;font-size:18px;}"
                "QCheckBox:hover{background-color:rgb(240, 248, 255);border-color:rgb(100, 149, 237);font-size:18px;}"
            )
            h_layout.addWidget(self.checkboxes[i])

        self.input_path_label = QTLabel("input_path_label", "Male_window", ">> 選擇輸入目錄:")
        self.input_path = QTButton("input_path", "Male_window")
        self.input_path.set_text("瀏覽")

        self.output_path_label = QTLabel("output_path_label", "Male_window", ">> 選擇輸出目錄:")
        self.output_path = QTButton("output_path", "Male_window")
        self.output_path.set_text("瀏覽")

        self.start_btn = QTButton("start", "Male_window")
        self.start_btn.set_text("Start")

        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.setStyleSheet(
            "QProgressBar{border: 2px solid gray; border-radius: 5px; background-color: #FFFFFF}"
            "QProgressBar::chunk{background-color: #87CEFA; width: 10px}"
            "QProgressBar{border: 2px solid gray; border-radius: 5px; text-align: center;}"
        )
        self.progress_label = QTLabel("progress_label", "Male_window", "目前進度")
        self.progress_label.label.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
        self.progress_label.label.hide()

        # self.clust = QtWidgets.QLineEdit()
        # self.clust.setFont(self.font)
        # self.clust.setObjectName("clust")
        # self.clust.setStyleSheet("#clust{border:2px groove gray;border-radius:10px;padding:2px 4px;font-size:18px;}")
        # self.clust.setPlaceholderText("分群數目:")
        # self.clust.setValidator(QtGui.QIntValidator(1, 100))

        grid_layout.addWidget(self.season_label.label, 0, 0)
        grid_layout.addWidget(self.seasons, 1, 0)
        grid_layout.addWidget(self.feature_label.label, 2, 0)
        grid_layout.addLayout(h_layout, 3, 0)
        grid_layout.addWidget(self.input_path_label.label, 4, 0)
        grid_layout.addWidget(self.input_path.pushButton, 5, 0)
        grid_layout.addWidget(self.output_path_label.label, 6, 0)
        grid_layout.addWidget(self.output_path.pushButton, 7, 0)
        # grid_layout.addWidget(self.clust, 8, 0)
        grid_layout.addWidget(self.progressBar, 8, 0)
        grid_layout.addWidget(self.progress_label.label, 9, 0)
        grid_layout.addWidget(self.start_btn.pushButton, 9, 0)
        grid_layout.setRowStretch(9, 1)
        self.setLayout(grid_layout)
        self.setConnect()
        self.onSeasonSelect()

    def setConnect(self):
        self.seasons.currentIndexChanged.connect(self.onSeasonSelect)
        self.checkboxes[0].clicked.connect(lambda: self.onFeatureSelect(0))
        self.checkboxes[1].clicked.connect(lambda: self.onFeatureSelect(1))
        self.checkboxes[2].clicked.connect(lambda: self.onFeatureSelect(2))
        self.input_path.pushButton.clicked.connect(lambda: self.onFolderSelect(True))
        self.output_path.pushButton.clicked.connect(lambda: self.onFolderSelect(False))
        self.start_btn.pushButton.clicked.connect(self.onStartCilcked)

    def onSeasonSelect(self):
        self.controller.OnSeasonsSelect()

    def onFeatureSelect(self, i):
        self.controller.OnFeatureSelect(i)

    def onFolderSelect(self, flag):
        self.controller.OnInputFolderSelect(flag)

    def setFolder(self, text, flag):
        if flag:
            self.input_path.pushButton.setText(self._translate("Male_window", text))
        else:
            self.output_path.pushButton.setText(self._translate("Male_window", text))

    def onStartCilcked(self):
        self.controller.OnStartClicked()

    def notify(self, text):
        QMessageBox.question(self, '錯誤', text, QMessageBox.Retry, QMessageBox.Retry)

    def start_clustering(self):

        self.set_UI_Enable(False)
        self.start_btn.pushButton.hide()
        self.progress_label.label.show()

    def restart(self):
        self.main_window.restart()

    def set_UI_Enable(self, bool):
        self.seasons.setEnabled(bool)
        self.checkboxes[0].setEnabled(bool)
        self.checkboxes[1].setEnabled(bool)
        self.checkboxes[2].setEnabled(bool)
        self.input_path.pushButton.setEnabled(bool)
        self.output_path.pushButton.setEnabled(bool)
        # self.clust.setEnabled(bool)
