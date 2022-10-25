import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
from glob import glob
Image.MAX_IMAGE_PIXELS = None


def find_image(folder, images):
    fTree = os.walk(folder, topdown=True)
    img_filename = ['jpg', 'tif', 'tiff', 'png', 'bmp', 'jpeg', 'jfif']
    image = []
    for f in img_filename:
        img = glob(folder + f'/*.{f}')
        image += img
    for img in image:
        try:
            tmp = Image.open(img)
            tmp.close()
            images.append(img)
        except:
            pass
    for dirs, subdirs, files in fTree:
        if len(subdirs) != 0:
            for subdir in subdirs:
                find_image(os.path.join(folder, subdir), images)
    return images


class IMAGE_Dataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        pil_image = Image.open(self.img_path[index]).convert('RGB')
        pil_image = pil_image.resize((1024, 1024))
        image = self.transform(pil_image)
        return image, self.img_path[index], np.asarray(pil_image)


class Partial_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.x = []
        self.y = []
        self.transform = transform
        self.num_classes = 0
        # print(self.root_dir.name)
        for i, _dir in enumerate(self.root_dir.glob('*')):
            count = 0
            j = 0
            for file in _dir.glob('*'):
                count += 1
            for file in _dir.glob('*'):
                if j < count/15:
                    self.x.append(file)
                    self.y.append(i)
                    j+=1
                    # print(file)
                    # input()
                else:
                    continue

            self.num_classes += 1
            # print(self.num_classes)
        # print(self.num_classes)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, self.y[index]


class Extract_Data(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.img = []
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([transforms.Resize((254, 254)), transforms.ToTensor(
            ), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        for i, img_file in enumerate(self.root_dir.glob('*')):
            self.img.append(img_file)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        image = Image.open(self.img[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image
