# -*- coding:utf-8 -*-
# Author : xuan
# Date : 2019/11/3 17:44
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import torchvision
from tqdm import tqdm
from matplotlib import pyplot as plt
import os


data_split = np.load(r"C:\Users\xuan\Desktop\教评系统分类代码\segmentation_data_split.npz")
print("train_x.shape, train_y.shape: ", data_split["train_x"].shape, data_split["train_y"].shape)
print("test_x.shape, test_y.shape: ", data_split["test_x"].shape, data_split["test_y"].shape)


class FixedFeatureDataset(Dataset):
    def __init__(self, data_split, train=True):
        super(FixedFeatureDataset, self).__init__()
        self.data_split = data_split
        self.train = train
        if self.train:
            self.train_x = self.data_split["train_x"]
            self.train_y = self.data_split["train_y"]
        else:
            self.test_x = self.data_split["test_x"]
            self.test_y = self.data_split["test_y"]

    def __getitem__(self, item):
        if self.train:
            return self.train_x[item], self.train_y[item]
        return self.test_x[item], self.test_y[item]

    def __len__(self):
        if self.train:
            return len(self.train_x)
        return len(self.test_x)


def show_confMat(confusion_mat, classes_name, set_name, out_dir="."):
    """
    可视化混淆矩阵，保存png格式
    :param confusion_mat: nd-array
    :param classes_name: list,各类别名称
    :param set_name: str, eg: 'valid', 'train'
    :param out_dir: str, png输出的文件夹
    :return:
    """
    # 归一化
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes_name)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes_name)))
    plt.xticks(xlocations, classes_name, rotation=90)
    plt.yticks(xlocations, classes_name)
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    plt.savefig(os.path.join('Confusion_Matrix_' + set_name + '.png'), bbox_inches="tight")
    plt.close()


trainData = FixedFeatureDataset(data_split, train=True)
testData = FixedFeatureDataset(data_split, train=False)

trainLoader = DataLoader(trainData, batch_size=4)
testLoader = DataLoader(testData, batch_size=4)

# net = torchvision.models.resnet50()
# net.fc = nn.Linear(2048, 6)

# net = torchvision.models.inception_v3()
# net.aux_logits = False
# net.transform_input = False
# net.fc = nn.Linear(2048, 6)

net = torchvision.models.densenet121()
net.classifier = nn.Linear(1024, 6)

model_path = r"D:\研究生生活\实验室\教室视频分析\运行结果\分类\教评分类结果\densenetMixUp_model\0.1\0.8785714285714286_densenet_9_classroomClassification.pkl"
state = torch.load(model_path)
net.load_state_dict(state["model_state"])
net.eval().cpu()
conf_mat = np.zeros([6, 6])
test_running_loss = 0.0
test_acc = 0.0
for data in tqdm(testLoader):
    img_feature, label = data
    img_feature = Variable(img_feature).cpu()
    net.eval()
    img_feature = Variable(img_feature).cpu()
    label = Variable(label).long().cpu()
    out = net(img_feature)
    pred = torch.max(out.data, 1)[1]
    test_acc += torch.sum(pred == label).item()
    for i in range(len(label)):
        true_i = np.array(label[i])
        pred_i = np.array(pred[i])
        conf_mat[true_i, pred_i] += 1.0

print(conf_mat)
class_names = ["look_student_point", "look_blackboard_point", "only_look_blackboard", "only_look_student", "read_book", "others"]
show_confMat(conf_mat, class_names, "Densenet_Mixup0.1")
test_acc = test_acc / len(testData)
print(test_acc)
