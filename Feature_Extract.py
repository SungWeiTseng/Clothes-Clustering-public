import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from Feature_Extraction_and_Clustering.dataset import IMAGE_Dataset, Partial_Dataset
import numpy as np
import cv2
from Feature_Extraction_and_Clustering.network import RESNet, VIT32
from PIL import ImageFile, Image

from preprocess.mask_rcnn import MRCNN

ImageFile.LOAD_TRUNCATED_IMAGES = True

# import configure as cfg

# CUDA_DEVICES = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# num_classes = 10
DATASET_ROOT = '/home/TSSW/Project/apparel/data/other'
MATERIAL_PATH_TO_WEIGHTS = 'material and texture model/M_model-1.00-best_train_acc.pth'
TEXTURE_PATH_TO_WEIGHTS = 'material and texture model/T_model-1.00-best_train_acc.pth'


def pil2cv(pilimg):
    img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return img


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def color_extract(images_path):
    color_feature = None
    maskedCNN = MRCNN()

    mask_data_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor()])
    dataset = IMAGE_Dataset(images_path, mask_data_transform)
    data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=False, num_workers=8)

    for img, img_path, np_img in data_loader:
        img = img.to(device)
        mask_imgs, masks = maskedCNN.get_max_segment(img, np_img.numpy())
        for img, mask in zip(mask_imgs, masks):
            img = np.array(img)[:, :, ::-1].copy()
            # img = cv2.resize(pil2cv(img), (256, 256))
            # img = cv_imread(path)
            # Convert RGB to BGR
            # if len(img.shape) != 3:
            #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            mask = np.array(mask)
            hist = cv2.calcHist([img], [0, 1, 2], mask, [10, 10, 10], [0, 256, 0, 256, 0, 256])
            # color = cv2.calcHist([np_img], [0, 1, 2], None, [8, 8, 8], [-1, 1, -1, 1, -1, 1]).flatten()
            hist = hist / (img.shape[0] * img.shape[1])
            hist = np.reshape(hist, (-1, 1000))
            if color_feature is None:
                color_feature = hist
            else:
                color_feature = np.concatenate((color_feature, hist), axis=0)
    return color_feature


def feature_extract(images_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask_data_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor()])
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    # transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    train_set = IMAGE_Dataset(images_path, mask_data_transform)
    data_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=False, num_workers=8)

    model_material = RESNet(classes=10)
    model_texture = RESNet(classes=10)
    material_feature = torch.Tensor([])
    texture_feature = torch.Tensor([])
    model_material.load_state_dict(torch.load(MATERIAL_PATH_TO_WEIGHTS, map_location=device))
    model_texture.load_state_dict(torch.load(TEXTURE_PATH_TO_WEIGHTS, map_location=device))
    model_material = model_material.to(device)
    model_material.eval()
    model_texture = model_texture.to(device)
    model_texture.eval()
    # material = VIT32()

    maskedCNN = MRCNN()
    with torch.no_grad():
        for img, img_path, np_img in data_loader:
            imgs = torch.FloatTensor([])
            img = img.to(device)
            mask_img, _ = maskedCNN.get_max_segment(img, np_img.numpy())
            for img in mask_img:
                imgs = torch.cat((imgs, data_transform(img).unsqueeze(0)), dim=0)
            imgs = imgs.to(device)

            # outputs = model_material(imgs)[0]
            outputs = model_material(imgs)[0]
            features = outputs.view(imgs.size(0), -1)
            if torch.cuda.is_available():
                features = features.cpu()
            material_feature = torch.cat((material_feature, features), 0)

            outputs = model_texture(imgs)[0]
            features = outputs.view(-1, 2048)
            if torch.cuda.is_available():
                features = features.cpu()
            texture_feature = torch.cat((texture_feature, features.cpu()), 0)

            # for img in images:
            #     img = (img.numpy() * 255).astype(np.uint8)
            #     hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            #     # hist = hist / (img.shape[0] * img.shape[1])
            #     hist = np.reshape(hist, (-1, 512))
            #     print(np.sum(hist))
            #     if color_feature is None:
            #         color_feature = hist
            #     else:
            #         color_feature = np.concatenate((color_feature, hist), axis=0)
    # for path in images_path:
    #     img = cv_imread(path)
    #     # Convert RGB to BGR
    #     # img = open_cv_image[:, :, ::-1].copy()
    #     hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    #     # color = cv2.calcHist([np_img], [0, 1, 2], None, [8, 8, 8], [-1, 1, -1, 1, -1, 1]).flatten()
    #     hist = hist / (img.shape[0] * img.shape[1])
    #     hist = np.reshape(hist, (-1, 512))
    #     if color_feature is None:
    #         color_feature = hist
    #     else:
    #         color_feature = np.concatenate((color_feature, hist), axis=0)

    return material_feature.numpy(), texture_feature.numpy()
