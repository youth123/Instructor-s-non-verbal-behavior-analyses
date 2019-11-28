# 为行为分类提供训练数据，批量读取文件夹的图片进行分割
# 基于PASCAL VOC数据集的分割结果进行原图的分割，目的是将人完整的分割出来
# 需要修改的参数：模型位置、源文件夹位置、分割后数据保存的文件位置

import extractors
import os
from torchvision import transforms as tfs
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

# empty lists to hold labels
labels = []
classes = ['background', 'person', 'blackboard', 'chair',
           'platform', 'door', 'screen', 'desk', 'airconditioning']


class Conv2dLayer(nn.Module):
    ''' 2D convolution with batch norm and ReLU '''

    def __init__(self, in_channels, out_channels, kernel_size,
                 groups=1, stride=1, padding=0, dilation=1, bias=True,
                 batchnorm_opts={'eps': 1e-3, 'momentum': 3e-4}):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      groups=groups, stride=stride, padding=padding,
                      dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels, **batchnorm_opts),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class ClassroomSegmentation(nn.Module):

    class ResBackBone(nn.Module):
        def __init__(self, backend="resnet50", pretrained=True):
            super().__init__()
            self.feats = getattr(extractors, backend)(pretrained)

        def forward(self, x):
            feats, class_f, feats_low = self.feats(x)
            return feats, class_f, feats_low

    class ASPP(nn.Module):
        ''' Atrous spatial pyramid pooling module '''

        def __init__(self, in_channels, output_stride=16):
            super().__init__()

            if output_stride not in {8, 16}:
                raise ValueError('Invalid output_stride; Supported values: {8, 16}')
            dilation_factor = 1 if output_stride == 16 else 2

            self.aspp = nn.ModuleList([
                Conv2dLayer(in_channels, 256, kernel_size=1, dilation=1),
                Conv2dLayer(in_channels, 256, kernel_size=3,
                                   dilation=6 * dilation_factor, padding=6 * dilation_factor),
                Conv2dLayer(in_channels, 256, kernel_size=3,
                                   dilation=12 * dilation_factor, padding=12 * dilation_factor),
                Conv2dLayer(in_channels, 256, kernel_size=3,
                                   dilation=18 * dilation_factor, padding=18 * dilation_factor)])

            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Conv2d(in_channels, 256, kernel_size=1),
                nn.ReLU(inplace=True))

            self.output_conv = Conv2dLayer(256 * 4 + 256, 256, kernel_size=1)

        def forward(self, x):
            x_aspp = (aspp(x) for aspp in self.aspp)
            x_pool = self.global_avg_pool(x)
            x_pool = F.interpolate(x_pool, size=x.shape[2:4])
            feats = torch.cat((*x_aspp, x_pool), dim=1)
            feats = self.output_conv(feats)
            return feats

    class Decoder(nn.Module):

        def __init__(self, low_in_channels, num_classes):
            super().__init__()
            self.conv_low = Conv2dLayer(low_in_channels, 48, kernel_size=1)
            self.conv_logit = nn.Conv2d(48 + 256, num_classes, kernel_size=3, padding=1)

        def forward(self, feats, low_feats):
            low_feats = self.conv_low(low_feats)
            feats = F.interpolate(feats, size=low_feats.shape[2:4],
                                        mode='bilinear', align_corners=True)
            feats = torch.cat((feats, low_feats), dim=1)
            logits = self.conv_logit(feats)
            return logits

    def __init__(self, num_classes, deep_features_size=1024):
        super().__init__()

        self.backbone = ClassroomSegmentation.ResBackBone()
        self.aspp = ClassroomSegmentation.ASPP(in_channels=2048,
                                 output_stride=16)
        self.decoder = ClassroomSegmentation.Decoder(low_in_channels=256,
                                       num_classes=num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_in):
        x, x_cls, x_low = self.backbone(x_in)
        x = self.aspp(x)
        logits = self.decoder(x, x_low)
        logits = F.interpolate(logits, size=x_in.shape[2:4],
                                     mode='bilinear', align_corners=True)

        auxiliary = F.adaptive_max_pool2d(input=x_cls, output_size=(1, 1)).view(-1, x_cls.size(1))
        return logits, self.classifier(auxiliary)


if __name__ == '__main__':
    # 建立模型，模型是别人基于PASCAL VOC训练的Deeplabv3+分割模型
    model_path = r"D:\研究生生活\实验室\教室视频分析\运行结果\ClassroomDataset_pytorch\pytorch\models\2\0.8495504410817009_classroomDemo1.pkl"
    model = ClassroomSegmentation(num_classes=9)
    state = torch.load(model_path, map_location=lambda storage, loc:storage)
    class_iou = state["class_iou"]
    for k, v in class_iou.items():
        print("{}: {}".format(classes[k], v))
    model.load_state_dict(state["model_state"])
    model.eval().cpu()

    # 循环读取文件夹中的图片
    data_dir = r'D:\dataset\classroom_6\ActionsClassifyResizeRename\\'
    contents = os.listdir(data_dir)
    labels = [each for each in contents if os.path.isdir(data_dir + each)]
    print(labels)

    for name in labels:
        # join the training data path and each species training folder
        data_path = os.path.join(data_dir, name)
        fileNames = os.listdir(data_path) # 分类文件夹下的文件名

        # 获取当前的标签
        current_label = name
        im_tfs = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # 循环读取子文件夹中的图片
        for file in fileNames:
            image_path = os.path.join(data_path, file)
            print(image_path)
            orignal_im = Image.open(image_path)
            pre_processed = im_tfs(orignal_im)
            pred, _ = model(pre_processed.unsqueeze(0).cpu())
            pred = pred.max(1)[1].squeeze().cpu().data.numpy()
            for x in range(orignal_im.size[0]):
                for y in range(orignal_im.size[1]):
                    # PASCAL VOC中第15类是person，保留person，其余设置为
                    # 黑色
                    if (pred[y, x] == 0):
                        orignal_im.putpixel((x, y), (0, 0, 0))
            out_path = os.path.join("D:/dataset/classroom_6/ActionsClassifySegmentation/", name)
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            img_path = os.path.join(out_path, file)
            orignal_im.save(img_path)
