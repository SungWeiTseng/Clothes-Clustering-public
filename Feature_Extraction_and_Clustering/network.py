import numpy as np
import torch.nn as nn
import torch
import math
import torchvision.models as models
# from vit_pytorch import ViT
# from vit_pytorch.extractor import Extractor

from Feature_Extraction_and_Clustering.VIT import vit_b_32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VIT32:

    def __init__(self):
        self.model = ViT(
            image_size=384,
            patch_size=32,
            num_classes=1000,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.model.eval()
        # self.model = self.model.to(device)
        torch.load()
        self.feature_extractor = Extractor(vit=self.model, device=device)
        self.feature_extractor = self.feature_extractor.to(device)
        # nn.Sequential(*list(self.model.children())[:-1])

    def show(self):
        # print(self.feature_extractor)
        print(self.model)

    def extract(self, img):
        print(img.shape)
        feature = self.feature_extractor(img)
        # feature = self.model(img)
        return feature[1]


class SSD:
    def __init__(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
        self.ssd_model.to(device)
        self.ssd_model.eval()

    def detect(self, images):
        detections_batch = self.ssd_model(images)
        results_per_input = self.utils.decode_results(detections_batch)
        # [[bboxes], [classes], [confidence]]
        best_results_per_input = [self.utils.pick_best(results, 0.40) for results in results_per_input]
        # [[bbox, class, confidence], [bbox, class, confidence], ...,]

        results = [[[bbox, cls, confidence] for bbox, cls, confidence in zip(result[0], result[1], result[2])] for
                   result in best_results_per_input]
        return results


class RESNet(nn.Module):
    def __init__(self, classes):
        super(RESNet, self).__init__()
        # Reference : PyTorch - Best way to get at intermediate layers in VGG and ResNet?
        # https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/2
        self.feature_extractor = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])

        # edit output class
        # in_features = self.feature_extractor.fc.in_features
        self.fc = nn.Linear(in_features=2048, out_features=classes)

    def forward(self, x):
        intermediate = self.feature_extractor(x)
        intermediate = intermediate.view(x.size(0), -1)
        out = self.fc(intermediate)
        return intermediate, out


if __name__ == "__main__":
    vit32 = VIT32()
    vit32.show()
    data = np.load(
        '../material and texture model/R50_L_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz')
