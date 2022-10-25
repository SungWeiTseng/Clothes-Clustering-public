import warnings
warnings.simplefilter("ignore", UserWarning)
from multiprocessing import freeze_support
freeze_support()
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import model


class Controller:
    def __init__(self, view):
        self.view = view
        self.model = model.Model()
        self.season_root1 = str()
        self.season_root2 = str()
        self.output_root = str()

    def OnSeasonsSelect(self):
        self.model.OnSeaeonSelect(self.view.seasons)

    def OnFeatureSelect(self, index):
        self.model.OnFeatureSelect(index, self.view.checkboxes[index])

    def OnInputFolderSelect(self, flag):
        folder = self.model.getFolder(self.view, flag)
        self.view.setFolder(folder, flag)

    def OnStartClicked(self):
        if self.view.finish:
            self.view.restart()
            return
        mode = self.model.check(self.view)
        if mode != 0:
            self.view.notify(mode)
            return
        self.view.start_clustering()
        self.model.Start_Clustering(self.view)
