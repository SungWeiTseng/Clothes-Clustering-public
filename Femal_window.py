import warnings

warnings.simplefilter("ignore", UserWarning)
from multiprocessing import freeze_support

freeze_support()
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, time
from glob import glob
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from execute_female import Runthread
from base_UI import *


class Ui_MainWindow(object):
    # 座標(x, y, w, h)，左上角為基準
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 500)
        MainWindow.setStyleSheet("#MainWindow{border-image:url(background.jpg);}")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.seasons_list = ['春夏', '秋冬']
        self.features_list = ['Texture', 'Material', 'Color']
        self.area = ['高訂', '倫敦', '米蘭', '紐約', '巴黎']

        self.season_label = QTLabel("season_label", "Female_window", ">> 選擇季節:")
        self.feature_label = QTLabel("feature_label", "Female_window", ">> 選擇特徵:")

        self.seasons = QTComboBox("season_ComboBox", "Female_window", self.seasons_list)

        grid_layout = QGridLayout()
        feature_h_layout = QHBoxLayout()
        area_combine_h_layout = QHBoxLayout()
        area_name_layout = QVBoxLayout()
        area_path_layout = QVBoxLayout()
        area_select_layout = QVBoxLayout()
        # area_select_num_clust = QVBoxLayout()

        self.select_input_folder = QTButton("input_folder", "Female_window")
        self.select_output_folder = QTButton("output_folder", "Female_window")
        self.select_input_folder.set_text("選擇輸入資料夾")
        self.select_output_folder.set_text("選擇輸出資料夾")

        self.checkboxes = []
        for i, feature in enumerate(self.features_list):
            self.checkboxes.append(QCheckBox(feature, MainWindow))
            self.checkboxes[i].toggle()
            self.checkboxes[i].setStyleSheet(
                "QCheckBox{border:1px groove gray;border-radius:10px;padding:2px 4px;background-color:#F8F8FF;font-size:18px;}"
                "QCheckBox:hover{background-color:rgb(240, 248, 255);border-color:rgb(100, 149, 237);font-size:18px;}"
            )
            feature_h_layout.addWidget(self.checkboxes[i])

        self.area_entry = []
        for i, area in enumerate(self.area):
            self.area_entry.append({
                "name": QTLabel(f"area{i + 1}_name", "Female_window", area),
                "path": QTLineEdit(f"area{i + 1}", "Female_window"),
                "button": QTButton(f"area{i + 1}_btn", "Female_window"),
                "clust": QTLineEdit("clust", "Female_window")
            })
            self.area_entry[i]["button"].set_text("瀏覽")
            # self.area_entry[i]["clust"].lineEdit.setPlaceholderText("群數")
            # self.area_entry[i]["clust"].lineEdit.setValidator(QtGui.QIntValidator(1, 100))
            # self.area_entry[i]["clust"].lineEdit.setFixedWidth(70)

            area_name_layout.addWidget(self.area_entry[i]["name"].label)
            area_path_layout.addWidget(self.area_entry[i]["path"].lineEdit)
            area_select_layout.addWidget(self.area_entry[i]["button"].pushButton)
            # area_select_num_clust.addWidget(self.area_entry[i]["clust"].lineEdit)

        area_combine_h_layout.addLayout(area_name_layout)
        area_combine_h_layout.addLayout(area_path_layout)
        area_combine_h_layout.addLayout(area_select_layout)
        # area_combine_h_layout.addLayout(area_select_num_clust)
        area_combine_h_layout.setStretchFactor(area_path_layout, 3)

        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.setStyleSheet(
            "QProgressBar{border: 2px solid gray; border-radius: 5px; background-color: #FFFFFF}"
            "QProgressBar::chunk{background-color: #87CEFA; width: 10px}"
            "QProgressBar{border: 2px solid gray; border-radius: 5px; text-align: center;}"
        )
        self.progress_label = QTLabel("progress_label", "Female_window", "目前進度")
        self.progress_label.label.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
        # self.progress_label.label.hide()

        self.start_btn = QTButton("start", "Female_window")
        self.restart_btn = QTButton("restart", "Female_window")
        self.restart_btn.set_text("Restart")
        self.start_btn.set_text("Start")

        grid_layout.addWidget(self.season_label.label, 0, 0)
        grid_layout.addWidget(self.seasons.ComboBox, 1, 0)
        grid_layout.addWidget(self.feature_label.label, 2, 0)
        grid_layout.addLayout(feature_h_layout, 3, 0)
        grid_layout.addWidget(self.select_input_folder.pushButton, 4, 0)
        grid_layout.addLayout(area_combine_h_layout, 5, 0)
        grid_layout.addWidget(self.select_output_folder.pushButton, 6, 0)
        grid_layout.addWidget(self.progressBar, 7, 0)
        grid_layout.addWidget(self.progress_label.label, 8, 0)
        grid_layout.addWidget(self.start_btn.pushButton, 9, 0)
        grid_layout.addWidget(self.restart_btn.pushButton, 9, 0)
        grid_layout.setRowStretch(8, 2)
        MainWindow.setLayout(grid_layout)


class Female_window(base_window):
    def __init__(self, main_window):
        super(Female_window, self).__init__()
        self.features_list = ['Texture', 'Material', 'Color']
        self.area = ['高訂', '倫敦', '米蘭', '紐約', '巴黎']
        self.area_folder_list = {'高訂': '', '倫敦': '', '米蘭': '', '紐約': '', '巴黎': ''}
        self.features = [True, True, True]
        self.select_season = None
        self.input_root = str()
        self.output_root = str()
        self.main_window = main_window
        self.done = False

        self.ui_main = Ui_MainWindow()
        self.ui_main.setupUi(self)
        self.ui_main.restart_btn.pushButton.hide()
        # main window initialize
        self.setWindowTitle(" 服飾分群 ")
        # set icon
        self.icon = QtGui.QIcon()
        self.icon.addPixmap(QtGui.QPixmap("icon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(self.icon)
        # fixed window
        self.setWindowFlags(QtCore.Qt.MSWindowsFixedSizeDialogHint)
        # 初始畫面按鍵功能
        self.ui_main.select_output_folder.pushButton.clicked.connect(self.output_browse)
        self.ui_main.start_btn.pushButton.clicked.connect(self.isStartClick)
        self.ui_main.restart_btn.pushButton.clicked.connect(self.initial)
        # self.ui_main.restart_btn.clicked.connect(self.initial)

        # 區域目錄選擇
        self.ui_main.area_entry[0]["button"].pushButton.clicked.connect(lambda: self.select_area(self.area[0]))
        self.ui_main.area_entry[1]["button"].pushButton.clicked.connect(lambda: self.select_area(self.area[1]))
        self.ui_main.area_entry[2]["button"].pushButton.clicked.connect(lambda: self.select_area(self.area[2]))
        self.ui_main.area_entry[3]["button"].pushButton.clicked.connect(lambda: self.select_area(self.area[3]))
        self.ui_main.area_entry[4]["button"].pushButton.clicked.connect(lambda: self.select_area(self.area[4]))
        # # 選擇根目錄
        self.ui_main.select_input_folder.pushButton.clicked.connect(self.select_input_root)
        # 選擇特徵
        self.ui_main.checkboxes[0].clicked.connect(lambda: self.OnFeatureSelect(0))
        self.ui_main.checkboxes[1].clicked.connect(lambda: self.OnFeatureSelect(1))
        self.ui_main.checkboxes[2].clicked.connect(lambda: self.OnFeatureSelect(2))
        # 選擇季節
        self.ui_main.seasons.ComboBox.currentIndexChanged.connect(self.OnSeaeonSelect)
        self.initial()

    # browse to choose input and output folder
    # if and else for memorizing the path which browse before

    def OnSeaeonSelect(self):
        self.select_season = "春夏" if self.ui_main.seasons.ComboBox.currentIndex() == 0 else "秋冬"

    def OnFeatureSelect(self, index):
        if self.ui_main.checkboxes[index].isChecked():
            self.features[index] = True
        else:
            self.features[index] = False

    def select_area(self, select_area):
        dest = os.path.join(os.path.expanduser("~"), 'Desktop')
        if self.area_folder_list[select_area] != "":
            dest = self.area_folder_list[select_area]
        path = QtWidgets.QFileDialog.getExistingDirectory(self, '選擇資料夾', dest)
        self.area_folder_list[select_area] = path
        for i, area in enumerate(self.area):
            if area == select_area:
                self.ui_main.area_entry[i]["path"].lineEdit.setText(self.area_folder_list[area])

    def select_input_root(self):
        root = QtWidgets.QFileDialog.getExistingDirectory(self, '選擇資料夾')
        self.input_root = root
        for path in glob(root + '/*'):
            for i, area in enumerate(self.area):
                if area in path:
                    self.area_folder_list[area] = path
                    self.ui_main.area_entry[i]["path"].lineEdit.setText(self.area_folder_list[area])

    def output_browse(self):
        if self.input_root != str():
            dest = self.input_root
        else:
            dest = os.path.join(os.path.expanduser("~"), 'Desktop')
        self.output_root = QtWidgets.QFileDialog.getExistingDirectory(self, '選擇資料夾', dest)
        self.ui_main.select_output_folder.pushButton.setText(self.output_root)

    # restart when the continue button is clicked
    def isRestartClick(self):
        # hide progressbar and initial the value
        # initial buttons and text box
        self.initial()

    # if dir is not exist, then create a dir
    def isDirExist(self, target_path):
        if not os.path.isdir(target_path):
            os.mkdir(target_path)

    def make_output_folder(self):
        self.output_root = os.path.join(self.output_root, "results")
        dir_name = ""
        for i, f in enumerate(self.features):
            if f:
                dir_name += f"{self.features_list[i]}" if len(dir_name) == 0 else f"_{self.features_list[i]}"
        idx = 0
        self.output_root = os.path.join(self.output_root, dir_name)
        save_path = self.output_root
        while os.path.isdir(save_path):
            idx += 1
            save_path = self.output_root + f'({idx})'
        self.output_root = save_path
        os.makedirs(self.output_root)

    # if main process finish, do:
    def finish(self):
        # set text
        global name_xlsx
        global name_dir
        string = f'分群完成 !\n結果紀錄於 {self.output_root}。\t'
        QMessageBox.question(self, '完成', string, QMessageBox.Ok, QMessageBox.Ok)
        self.ui_main.start_btn.set_text("restart")
        self.ui_main.restart_btn.pushButton.show()
        self.done = True
        # clean and init
        self.show()

    # start_login & call_backlog控制執行緒回傳
    def start_classify(self):
        input_path = list(self.area_folder_list.values())
        self.thread = Runthread(input_path, self.output_root, self.n_clusters, self.area, self.select_season,
                                self.features, self)
        self.thread._signal.connect(self.call_backlog)  # 執行緒連接回UI的動作
        self.thread.start()

    def call_backlog(self, msg):
        self.ui_main.progressBar.setValue(int(msg))
        if int(msg) == 100:
            self.finish()
            self.thread.quit()

    def set_input_enable(self, mode: bool):
        self.ui_main.seasons.ComboBox.setEnabled(mode)
        self.ui_main.checkboxes[0].setEnabled(mode)
        self.ui_main.checkboxes[1].setEnabled(mode)
        self.ui_main.checkboxes[2].setEnabled(mode)
        self.ui_main.select_input_folder.pushButton.setEnabled(mode)
        self.ui_main.select_output_folder.pushButton.setEnabled(mode)
        for area in self.ui_main.area_entry:
            # area["clust"].lineEdit.setEnabled(mode)
            area["path"].lineEdit.setEnabled(mode)
            area["button"].pushButton.setEnabled(mode)
        self.ui_main.start_btn.pushButton.setEnabled(mode)

        if mode:
            self.ui_main.start_btn.pushButton.show()
        else:
            self.ui_main.start_btn.pushButton.hide()

    def initial(self):

        if self.done:
            self.main_window.restart()
            return
        self.OnSeaeonSelect()

        # self.input_root = str()
        # self.output_root = str()
        # self.area_folder_list = {'高訂': '', '倫敦': '', '米蘭': '', '紐約': '', '巴黎': ''}
        # self.set_input_enable(True)
        # self.ui_main.select_output_folder.pushButton.text().clear()
        # self.ui_main.restart_btn.hide()

    #
    #     self.ui_main.lineEdit.clear()
    #     self.ui_main.lineEdit2.clear()
    #     self.ui_main.lineEdit3.clear()
    #     self.ui_main.lineEdit4.clear()
    #     self.ui_main.lineEdit5.clear()
    #     self.ui_main.lineEdit6.clear()
    #     self.ui_main.lineEdit7.clear()
    #     self.ui_main.lineEdit8.clear()
    #     self.ui_main.lineEdit9.clear()
    #     self.ui_main.lineEdit0.clear()
    #
    #     self.ui_main.clust0.clear()
    #     self.ui_main.clust1.clear()
    #     self.ui_main.clust2.clear()
    #     self.ui_main.clust3.clear()
    #     self.ui_main.clust4.clear()
    #     self.ui_main.clust5.clear()
    #     self.ui_main.clust6.clear()
    #     self.ui_main.clust7.clear()
    #     self.ui_main.clust8.clear()
    #     self.ui_main.clust9.clear()
    #     self.ui_main.label_3.setText("")
    #     self.ui_main.progressBar.hide()
    #     self.ui_main.progressBar.setValue(0)
    # # action when start button is clicked

    def isStartClick(self):
        # error messages
        flag = False
        for area in list(self.area_folder_list.values()):
            if area != "":
                flag = True
                break
        if not flag:
            QMessageBox.question(self, '錯誤', '請選擇輸入的資料夾。\t', QMessageBox.Retry, QMessageBox.Retry)
            self.show()
        if self.output_root == str():
            QMessageBox.question(self, '錯誤', '請選擇輸出的資料夾。\t', QMessageBox.Retry, QMessageBox.Retry)
            self.show()

        else:
            self.make_output_folder()
            # get number of cluster and check the value
            # self.n_clusters = [
            #     [self.ui_main.area_entry[].text(), self.ui_main.clust2.text(), self.ui_main.clust3.text(),
            #      self.ui_main.clust4.text(), self.ui_main.clust5.text()]]
            self.n_clusters = [area["clust"].lineEdit.text() for area in self.ui_main.area_entry]
            # check_length = [self.check_length(n) for s_clust in self.n_clusters for n in s_clust]

            if not (self.features[0] or self.features[1] or self.features[2]):
                QMessageBox.question(self, '錯誤', f'請至少選擇一項特徵。\t', QMessageBox.Retry, QMessageBox.Retry)
                self.show()
                return
            '''
            for i, path in enumerate(list(self.area_folder_list.values())):
                if path != "" and self.n_clusters[i] == "":
                    QMessageBox.question(self, '錯誤', f'請輸入分群數目。\t', QMessageBox.Retry, QMessageBox.Retry)
                    self.show()
                    return
                if path != "" and int(self.n_clusters[i]) < 2:
                    QMessageBox.question(self, '錯誤', f'分群數目不得小於2。\t', QMessageBox.Retry, QMessageBox.Retry)
                    self.show()
                    return
            '''
            # end checking, start main process
            self.set_input_enable(False)
            # self.ui_main.label_3.show()
            self.ui_main.progressBar.show()

            # start...
            try:
                time.sleep(0.1)
                self.start_classify()
            except Exception as e:
                print(e)
                QMessageBox.question(self, '錯誤', '輸入的資料有誤，請確認資料正確性。\t', QMessageBox.Retry,
                                     QMessageBox.Retry)
                self.show()
                self.isRestartClick()
