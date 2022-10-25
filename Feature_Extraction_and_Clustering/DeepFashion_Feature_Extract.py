import torch
from torchvision import transforms
from pathlib import Path
from torch.utils.data import DataLoader
from Feature_Extraction_and_Clustering.dataset import IMAGE_Dataset, Partial_Dataset
import numpy as np
import cv2
import glob
import os
import datetime
from Feature_Extraction_and_Clustering.network import RESNet
from PIL import Image
from PIL import ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# num_classes = 10
MATERIAL_PATH_TO_WEIGHTS = '../material and texture model/M_model-1.00-best_train_acc.pth'
TEXTURE_PATH_TO_WEIGHTS = '../material and texture model/T_model-1.00-best_train_acc.pth'


def get_path(img_root):
    root_dir = Path(img_root)
    path = []
    for i, _dir in enumerate(root_dir.glob('*')):
        for file in _dir.glob('*.jpg'):
            path.append(f'{file}')
            # print(path)
            # input()
    return path


def feature_extract(img_path, feature_type='color', save=True):
    """
    feature_type : material(2048), texture(2048), color(512)

    inputs : (img_root_dir, feature_type)
    outputs : features

    """
    # all_img_file = glob.glob(img_root+'/*/*.jpg')
    all_img_file = img_path

    if feature_type == 'material':
        State = 0
        model = RESNet(classes=10)
        model.load_state_dict(torch.load(MATERIAL_PATH_TO_WEIGHTS, map_location='cuda'))
        # model = model.cuda(CUDA_DEVICES)
        model.cuda()
        model.eval()
    elif feature_type == 'texture':
        State = 0
        model = RESNet(classes=10)
        model.load_state_dict(torch.load(TEXTURE_PATH_TO_WEIGHTS, map_location='cuda'))
        # model = model.cuda(CUDA_DEVICES)
        model.cuda()
        model.eval()
    elif feature_type == 'color':
        State = 1

    if not State:
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        train_set = IMAGE_Dataset(all_img_file, data_transform)
        data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=False, num_workers=0)
        with torch.no_grad():
            # start = datetime.datetime.now()
            all_features = torch.cuda.FloatTensor([])
            for i, (inputs, labels) in enumerate(tqdm(data_loader)):
                inputs = inputs.to('cuda')
                outputs = model(inputs)[0]
                features = outputs.view(-1, 2048)
                all_features = torch.cat((all_features, features), 0)

            all_features = all_features.cpu().numpy()

    else:
        # start = datetime.datetime.now()
        all_features = []
        '''
        data_transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        train_set = IMAGE_Dataset(all_img_file, data_transform)
        data_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=False, num_workers=8)

        ssd_model = SSD()
        
        all_features = []
        with torch.no_grad():
            for i, (images, img_path) in enumerate(tqdm(data_loader)):
                images = images.to(device)
                best_results_per_input = ssd_model.detect(images)
                for image_idx, results in enumerate(best_results_per_input):
                    bbox_left, bbox_top, bbox_right, bbox_bot = 1, 1, 0, 0
                    # bboxes, classes, confidences = results
                    count = 0
                    for idx, (bbox, cls, confidences) in enumerate(results):
                        if cls - 1 != 0:
                            continue
                        count += 1
                        left, bot, right, top = bbox
                        # 上下顛倒
                        if bbox_left > left:
                            bbox_left = left
                        if bbox_top > bot:
                            bbox_top = bot
                        if bbox_right < right:
                            bbox_right = right
                        if bbox_bot < top:
                            bbox_bot = top
                    if count == 0:
                        bbox_left, bbox_top, bbox_right, bbox_bot = 0, 0, 1, 1
                    crop_image = Image.open(img_path[image_idx]).convert('RGB')
                    width, height = crop_image.size
                    crop_image = crop_image.crop(
                        (bbox_left * width, bbox_top * height, bbox_right * width, bbox_bot * height))
                    open_cv_image = np.array(crop_image)
                    # Convert RGB to BGR
                    img = open_cv_image[:, :, ::-1].copy()
                    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])  # 512 dim
                    hist = hist/(img.shape[0]*img.shape[1])
                    all_features.append(hist.flatten())
        '''
        for path in img_path:
            img = Image.open(path).convert('RGB')
            open_cv_image = np.array(img)
            # Convert RGB to BGR
            img = open_cv_image[:, :, ::-1].copy()
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])  # 512 dim
            hist = hist / (img.shape[0] * img.shape[1])
            all_features.append(hist.flatten())
        all_features = np.asarray(all_features, dtype=np.float32)
        all_features = np.reshape(all_features, (-1, 512))

    return all_features


def partial_feature_extract(img_root, feature_type='color', save=True):
    """
    feature_type : material(2048), texture(2048), color(512)

    inputs : (img_root_dir, feature_type)
    outputs : features

    """
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    train_set = Partial_Dataset(Path(DATASET_ROOT), data_transform)
    data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=False, num_workers=0)

    if feature_type == 'material':
        State = 0
        model = RESNet(classes=10)
        model.load_state_dict(torch.load(MATERIAL_PATH_TO_WEIGHTS))
        # model = model.cuda(CUDA_DEVICES)
        model.cuda()
        model.eval()
    elif feature_type == 'texture':
        State = 0
        model = RESNet(classes=10)
        model.load_state_dict(torch.load(TEXTURE_PATH_TO_WEIGHTS))
        # model = model.cuda(CUDA_DEVICES)
        model.cuda()
        model.eval()
    elif feature_type == 'color':
        State = 1

    partial_img_file = Partial_Dataset(Path(DATASET_ROOT))
    partial_features = torch.Tensor([]).cuda()
    judge = 0

    if not State:
        with torch.no_grad():
            start = datetime.datetime.now()
            for i, (inputs, labels) in enumerate(data_loader):
                inputs = inputs.cuda()
                # labels = labels.to(device)
                outputs = model(inputs)[0]
                features = outputs.view(-1, 2048)
                # print(features.size())

                if judge == 0:
                    partial_features = features
                    judge += 1
                else:
                    partial_features = torch.cat((partial_features, features), 0)

                if i % 20 == 0 and (i != 0):
                    print(f'i = {i}')
            
            partial_features = partial_features.cpu().numpy()
            np.save(f"{FEATURES_ROOT}/one-fifteenth_{feature_type}.npy", partial_features)
            print(f'oone-fifteenth {feature_type} is saved, size: {partial_features.shape}')

            end = datetime.datetime.now()
            print(f'{feature_type} using time : {end - start}')
                

            # print(all_features.size())
            # all_features = all_features.cpu().numpy()

                # if index % cfg.LOG_INTERVAL == 0 and (index != 0):
                #     print(f'{index} / {len(all_img_file)} ({(index/len(all_img_file))*100 : .2f} %)')

    else:
        start = datetime.datetime.now()
        partial_features = []
        img_folder_dir = glob.glob(DATASET_ROOT + '/*')

        for name in img_folder_dir:
            img_dir = glob.glob(name + '/*.jpg')
            #print(name)
            num = len([img_dir for img_dir in os.listdir(name) if os.path.isfile(os.path.join(name, img_dir))])
            count = 0
            
            for img_name in img_dir:
                if count < num/15: #the top 1/15
                    img = cv2.imread(img_name)
                    hist = cv2.calcHist([img], [0, 1, 2],None, [8, 8, 8], [0, 256, 0, 256, 0, 256]) # 512 dim
                    #print ("3D histogram shape: %s , with %d values" % (hist.shape, hist.flatten().shape[0]))
                    #print ("image shape :", img.shape)
                    hist = hist/(img.shape[0]*img.shape[1])
                    print(hist.shape)
                    partial_features.append(hist.flatten()) 
                    count += 1
                    #print(count, "/", num)
                else:
                        pass 

            # if index % cfg.LOG_INTERVAL == 0 and (index != 0):
            #     print(
            #         f'{index} / {len(all_img_file)} ({(index/len(all_img_file))*100 : .2f} %)')
        
        partial_features = np.asarray(partial_features, dtype=np.float32)
        partial_features = np.reshape(partial_features, (-1, 512))
        np.save(f"{FEATURES_ROOT}/one-fifteenth_{feature_type}.npy", partial_features)
        print(f'one-fifteenth {feature_type} is saved, size: {partial_features.shape}')

        end = datetime.datetime.now()
        print(f'{feature_type} using time : {end - start}')

    # if save:
    #     np.save(f"{FEATURES_ROOT}/{feature_type}.npy", all_features)
    #     print(f'{feature_type} is saved, size: {all_features.shape}')

    if feature_type != 'color':
        del model

    return partial_features


if __name__ == '__main__':
    feature_extract(['D:/時裝週原始資料-2021/女裝/春夏/04米蘭/Boss_166/BOSS, 22SS (69).JPG', 'D:/時裝週原始資料-2021/女裝/春夏/04米蘭/Boss_166/BOSS, 22SS (69).JPG'])