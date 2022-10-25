import torch
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.patches as patches
from torchvision import transforms
import numpy as np


def detected(batch_img):
    ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ssd_model.to('cuda')
    ssd_model.eval()

    # data_transform = transforms.Compose([
    #     # transforms.Resize((300, 300)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    #
    # input = Image.open(img_path).convert('RGB').resize((300, 300))
    # numpy_img = np.array(input)

    # tensor_image = data_transform(input).view(1, 3, 300, 300)
    # tensor_image = tensor_image.to(device)

    with torch.no_grad():
        batch_img = batch_img.to(device)
        detections_batch = ssd_model(batch_img)
        results_per_input = utils.decode_results(detections_batch)
        best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]
        classes_to_labels = utils.get_coco_object_dictionary()

    for image_idx in range(len(best_results_per_input)):
        fig, ax = plt.subplots(1)
        # Show original, denormalized image...
        # image = numpy_img / 2 + 0.5
        ax.imshow(numpy_img)
        # ...with detections
        bboxes, classes, confidences = best_results_per_input[image_idx]
        for idx in range(len(bboxes)):
            if classes[idx] - 1 != 0:
                continue
            left, bot, right, top = bboxes[idx]
            x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx] * 100),
                    bbox=dict(facecolor='white', alpha=0.5))
    plt.show()


if __name__ == '__main__':
    # ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
    # utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ssd_model.to('cuda')
    ssd_model.eval()

    data_transform = transforms.Compose([
        # transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    input = Image.open("D:/時裝週原始資料-2021/女裝/春夏/02高訂/Alexandre Vauthier_134/Alexandre Vauthier Couture Spring 2022 (1).JPG").convert('RGB').resize((300, 300))
    numpy_img = np.array(input)
    tensor_image = data_transform(input).view(1, 3, 300, 300)
    tensor_image = tensor_image.to(device)

    with torch.no_grad():
        detections_batch = ssd_model(tensor_image)
        results_per_input = utils.decode_results(detections_batch)
        best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]
        classes_to_labels = utils.get_coco_object_dictionary()

    for image_idx in range(len(best_results_per_input)):
        fig, ax = plt.subplots(1)
        # Show original, denormalized image...
        # image = numpy_img / 2 + 0.5
        ax.imshow(numpy_img)
        # ...with detections
        bboxes, classes, confidences = best_results_per_input[image_idx]
        for idx in range(len(bboxes)):
            if classes[idx] - 1 != 0:
                continue
            left, bot, right, top = bboxes[idx]
            x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx] * 100),
                    bbox=dict(facecolor='white', alpha=0.5))
    plt.show()
