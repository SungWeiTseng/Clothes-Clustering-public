import os
import torch
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from Feature_Extraction_and_Clustering.dataset import find_image


class MRCNN:
    def __init__(self):
        # self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model = torch.load("material and texture model/mask.pth")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        self.model.eval()
        self.COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def get_prediction(self, tensor_img, threshold):
        batch_pred = self.model(tensor_img)
        pred_score = [list(pred['scores'].cpu().detach().numpy()) for pred in batch_pred]
        pred_t = []
        for score in pred_score:
            t = [score.index(x) for x in score if x > threshold]
            pred_t.append(t[-1] if len(t) > 0 else -1)

        masks = [(pred['masks'] > 0.5).view(-1, tensor_img.shape[2], tensor_img.shape[3]).cpu().detach().numpy()[:thr + 1] for pred, thr in zip(batch_pred, pred_t)]
        classes = [[self.COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred['labels'].cpu().detach().numpy())][:thr + 1] for pred, thr in zip(batch_pred, pred_t)]
        boxes = [[[(i[0], i[1]), (i[2], i[3])] for i in list(pred['boxes'].cpu().detach().numpy())][:thr + 1] for pred, thr in zip(batch_pred, pred_t)]

        return masks, boxes, classes

    def get_prediction_for_one(self, img_path, threshold):
        img = Image.open(img_path)
        transform = T.Compose([T.ToTensor()])
        img = transform(img)
        pred = self.model([img])
        # print('pred')
        # print(pred)
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        # print("masks>0.5")
        # print(pred[0]['masks'] > 0.5)
        masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
        # print("this is masks")
        # print(masks)
        pred_class = [self.COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
        masks = masks[:pred_t + 1]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]
        return masks, pred_boxes, pred_class

    def random_colour_masks(self, image):
        colours = [[255, 255, 255], [255, 255, 255], [255, 255, 255]]
        # colours = [[25, 25, 255], [25, 25, 25], [25, 25, 25]]

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        r[image == 1], g[image == 1], b[image == 1] = colours[0]
        coloured_mask = np.stack([r, g, b], axis=2)
        return coloured_mask

    def create_dir(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")

    def instance_segmentation_api_orig(self, img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
        masks, boxes, pred_cls = self.get_prediction(img_path, threshold)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(len(masks)):
            rgb_mask = self.random_colour_masks(masks[i])
            img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
            cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)

        plt.figure(figsize=(20, 30))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def get_max_segment(self, tensor_img, np_img, threshold=0.5):
        batch_masks, batch_bboxes, batch_pred_cls = self.get_prediction(tensor_img, threshold)
        is_person = [[cls in 'person' for cls in cls_per_img] for cls_per_img in batch_pred_cls]

        max_batch_mask = []
        max_batch_box = []
        for classes, bboxes, masks in zip(is_person, batch_bboxes, batch_masks):
            if len(classes) == 0:
                max_batch_box.append([])
                max_batch_mask.append([])
                continue
            np_bboxes = np.asarray(bboxes)[classes, ...]
            np_masks = np.asarray(masks)[classes, ...]
            if len(np_bboxes) == 0:
                max_batch_box.append([])
                max_batch_mask.append([])
                continue
            max_idx = np.argmax(np.sum(np_bboxes[:, 1] - np_bboxes[:, 0], 1))
            max_batch_box.append(np_bboxes[max_idx])
            max_batch_mask.append(np_masks[max_idx])

        ret_imgs = []
        ret_mask = []
        for mask, bbox, img in zip(max_batch_mask, max_batch_box, np_img):
            if len(mask) == 0:
                masked_img = Image.fromarray(img)
                ret_imgs.append(masked_img)
                mask = np.ones((img.shape[0], img.shape[1])).astype(np.uint8)
                ret_mask.append(Image.fromarray(mask))
                continue
            mask1c = (mask * 255).astype(np.uint8)
            rgb_mask = self.random_colour_masks(mask)
            masked_img = cv2.bitwise_and(img, rgb_mask)
            masked_img = Image.fromarray(masked_img)
            masked_img = masked_img.crop((*bbox[0], *bbox[1]))

            mask1c = Image.fromarray(mask1c)
            mask1c = mask1c.crop((*bbox[0], *bbox[1]))
            ret_imgs.append(masked_img)
            ret_mask.append(mask1c)
            # masked_img.save(save_path + "/" + "{}_masked.jpg".format("123"))
        return ret_imgs, ret_mask


class IMAGE_Dataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        pil_image = Image.open(self.img_path[index]).convert('RGB')
        row_size_w = pil_image.size[0]
        row_size_h = pil_image.size[1]
        pil_image = pil_image.resize((1024, 1024))
        image = self.transform(pil_image)
        return image, self.img_path[index], np.asarray(pil_image), row_size_w, row_size_h


if __name__ == '__main__':
    maskedCNN = MRCNN()
    # img = Image.open('D:/時裝週原始資料-2021/女裝/秋冬/06巴黎/中間有一個斜 VictoriaTomas, 21AW/VictoriaTomas, 21AW (24).JPG')
    # np_img = np.asarray(img)
    # transform = T.Compose([T.ToTensor()])
    # img = transform(img).unsqueeze(0)
    # img = maskedCNN.get_max_segment(img, np.asarray([np_img]))
    # img[0].show()

    # fold = "D:/時裝週原始資料-2021"
    fold = "/mnt/Nami/TSSW/時裝週原始資料-2021/男裝/春夏/4SDesigns"
    images_path = find_image(fold, [])
    data_transform = T.Compose([T.Resize((1024, 1024)), T.ToTensor()])
    # data_transform = T.Compose([T.ToTensor()])
    dataset = IMAGE_Dataset(images_path, data_transform)
    data_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, num_workers=4)

    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

    for img, img_path, np_img, row_size_w, row_size_h in tqdm.tqdm(data_loader):
        img = img.to(device)
        mask_img = maskedCNN.get_max_segment(img, np_img.numpy())
        for img, path, size_w, size_h in zip(mask_img, img_path, row_size_w, row_size_h):
            save_name = path.replace("時裝週原始資料-2021", "時裝週原始資料-2021_segment")
            filename = save_name.split("/")[-1]
            save_dir = save_name.replace(f"/{filename}", "")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            bbox_size = img.size
            resize_w = size_w.item() / 1024 * img.size[0]
            resize_h = size_h.item() / 1024 * img.size[1]
            scale = 256.0 / min(resize_w, resize_h)
            img = img.resize((int(resize_w * scale), int(resize_h * scale)))
            img.save(save_name)

