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
from torchvision import models
from tensorboardX import SummaryWriter
import datetime
from torch.nn import functional as F
from metrics import runningScore

data_root = r"/home/jil/dataset/ClassroomDataset/ClassroomDataset"
device = torch.device("cuda:0")

def read_image(root=data_root, train=True):
    txt_fname = root + "/Segmentation/5/" + ("train.txt" if train else "val.txt")
    with open(txt_fname, "r") as f:
        images = f.read().split()
    data = [os.path.join(root, "JPEGImages", i+".jpg") for i in images]
    label = [os.path.join(root, "SegmentationClassPNG", i+".png") for i in images]
    return data, label

data, label = read_image()

def rand_crop(data, label, height, width):
    data = np.array(data)
    label = np.array(label)
    h, w, _ = data.shape
    top = random.randint(0, h - height)
    left = random.randint(0, w - width)
    data = data[top:top+height, left:left+width]
    label = label[top:top+height, left:left+width]
    return data, label


classes = ['background', 'person', 'blackboard', 'chair',
           'platform', 'door', 'screen', 'desk', 'airconditioning']

# RGB color for each class
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0]]


cm2lbl = np.zeros(256**3)
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1]) * 256 + cm[2]] = i

def image2label(im):
    data = np.array(im, dtype="int32")
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype="int64")

label_im = Image.open(label[0]).convert("RGB")
label = image2label(label_im)
print(label[150:160, 240:250])

def img_transforms(im, label, crop_size):
    im, label = rand_crop(im, label, *crop_size)
    im_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    im = im_tfs(im)
    label = image2label(label)
    label = torch.from_numpy(label)
    return im, label


class DataPrefetcher(object):
    def __init__(self, loader, device, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None
        self.device = device

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_cls = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_cls = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(device=self.device, non_blocking=True)
            self.next_target = self.next_target.cuda(device=self.device, non_blocking=True)
            self.next_cls = self.next_cls.cuda(device=self.device, non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            cls = self.next_cls
            self.preload()
            count += 1
            yield input, target, cls
            if type(self.stop_after) is int and (count > self.stop_after):
                break



class ClassroomSegDataset(Dataset):
    def __init__(self, train, crop_size, transforms):
        self.crop_size = crop_size
        self.transforms = transforms
        data_list, label_list = read_image(train=train)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)
        print("Read " + str(len(self.data_list)) + " images")

    def _filter(self, images):
        return [
            im for im in images if (
                Image.open(im).size[1] >= self.crop_size[0] and
                Image.open(im).size[0] >= self.crop_size[1]
            )
        ]

    def __getitem__(self, item):
        img = self.data_list[item]
        label = self.label_list[item]
        img = Image.open(img)
        label = Image.open(label).convert("RGB")
        img, label = self.transforms(img, label, self.crop_size)
        cls = self.getcls(label)
        return img, label, cls

    def getcls(self,lbl):
        vector_lbl = lbl.contiguous().view(1,-1).numpy()
        set_lbl = set(np.squeeze(vector_lbl,0))
        cls = np.zeros((9),dtype=int)
        for s in set_lbl:
            cls[int(s)] = 1
        cls = torch.from_numpy(np.array(cls)).long()
        return cls

    def __len__(self):
        return len(self.data_list)

input_shape = (224, 224)
voc_train = ClassroomSegDataset(True, input_shape, img_transforms)
voc_test = ClassroomSegDataset(False, input_shape, img_transforms)

train_data = DataLoader(voc_train, 4, shuffle=True, num_workers=4)
valid_data = DataLoader(voc_test, 4, num_workers=4)

cuda_train_data = DataPrefetcher(train_data, device=device)
cuda_valid_data = DataPrefetcher(valid_data, device=device)


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



max_epochs = 80
num_classes = 9
net = ClassroomSegmentation(num_classes=num_classes)
net = net.cuda()
seg_criterion = nn.NLLLoss2d()
cls_criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-4)
running_metrics_train = runningScore(num_classes)
running_metrics_val = runningScore(num_classes)
best_iou = -100.0

for e in range(max_epochs):
    train_loss = 0

    prev_time = datetime.datetime.now()
    net = net.train()
    print("epoch " + str(e))
    for data in cuda_train_data:
        im = Variable(data[0].cuda())
        label = Variable(data[1].cuda())
        in_cls = Variable(data[2].cuda()).float()

        out, out_cls = net(im)
        out = F.log_softmax(out, dim=1)
        seg_loss = seg_criterion(out, label)
        cls_loss = cls_criterion(out_cls, in_cls)
        loss = seg_loss + 0.4 * cls_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data.item()

        label_pred = out.max(dim=1)[1].data.cpu().numpy()
        label_true = label.data.cpu().numpy()
        running_metrics_train.update(label_true, label_pred)

    score, class_iou = running_metrics_train.get_scores()
    for k, v in score.items():
        print("{}: {}".format(k, v))


    for k, v in class_iou.items():
        print("{}: {}".format(classes[k], v))


    cur_time = datetime.datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    epoch_str = ('Epoch: {}, Train Loss: {:.5f} '.format(e, train_loss / len(train_data)))
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(epoch_str + time_str)
    running_metrics_train.reset()

    if e % 1 == 0:
        net = net.eval()
        eval_loss = 0

        for data in cuda_valid_data:
            im = Variable(data[0].cuda(), volatile=True)
            label = Variable(data[1].cuda(), volatile=True)
            in_cls = Variable(data[2].cuda(), volatile=True).float()
            out, out_cls = net(im)
            out = F.log_softmax(out, dim=1)
            seg_loss = seg_criterion(out, label)
            #cls_loss = cls_criterion(out_cls, in_cls)
            loss = seg_loss #+ 0.4 * cls_loss
            eval_loss += loss.data.item()

            label_pred = out.max(dim=1)[1].data.cpu().numpy()
            label_true = label.data.cpu().numpy()
            running_metrics_val.update(label_true, label_pred)

        print("============eval===============")
        score, class_iou = running_metrics_val.get_scores()
        for k, v in score.items():
            print("{}: {}".format(k, v))


        for k, v in class_iou.items():
            print("{}: {}".format(classes[k], v))


        epoch_str = ('Valid Loss: {:.5f} '.format(eval_loss / len(valid_data)))
        print(epoch_str)
        print("============eval===============")
        running_metrics_val.reset()

        if score["Mean IoU : \t"] >= best_iou:
            best_iou = score["Mean IoU : \t"]
            state = {
                "epoch": i + 1,
                "model_state": net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_iou": best_iou,
                "class_iou": class_iou
            }
            save_path = "/home/jil/classroom_models/5/{}_classroomDemo1.pkl".format(str(best_iou))
            torch.save(state, save_path)





