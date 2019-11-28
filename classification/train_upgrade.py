#-*- coding:utf-8 -*-
# Author : xuan
# Date : 2019/11/3 17:44
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
from torch import optim
import torchvision
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.nn import functional as F
import time
import copy
import argparse
import os
import csv

parser = argparse.ArgumentParser("please enter the backbone: resnet, inception, densenet")
parser.add_argument("backbone", choices=["resnet", "inception", "densenet"])


data_split = np.load("segmentation_data_split.npz")
print("train_x.shape, train_y.shape: ", data_split["train_x"].shape, data_split["train_y"].shape)
print("test_x.shape, test_y.shape: ", data_split["test_x"].shape, data_split["test_y"].shape)


def load_weights_sequential(target, source_state):
    model_to_load = {k: v for k, v in source_state.items() if k in target.state_dict().keys()}
    # print(model_to_load.keys())
    target.load_state_dict(model_to_load, strict=False)


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


trainData = FixedFeatureDataset(data_split, train=True)
testData = FixedFeatureDataset(data_split, train=False)

trainLoader = DataLoader(trainData, batch_size=4)
testLoader = DataLoader(testData, batch_size=4)

args = parser.parse_args()
total_best_acc = []
total_train_epoch_time = []
total_test_epoch_time = []

model_dir = args.backbone + "_model"

for exp_i in range(5):
    print("*****************************************")
    exp_i = exp_i + 1
    print("exp_i: ", exp_i)

    # "resnet", "inception", "densenet"
    if args.backbone == "resnet":
        print("resnet")
        net = torchvision.models.resnet50(pretrained=True)
        net.fc = nn.Linear(2048, 6)
    elif args.backbone == "inception":
        print("inception")
        net = torchvision.models.inception_v3(pretrained=True)
        net.aux_logits = False
        net.transform_input = False
        net.fc = nn.Linear(2048, 6)
    else:
        print("densenet")
        net = torchvision.models.densenet121(pretrained=True)
        net.classifier = nn.Linear(1024, 6)

    net.cuda()
    #net = nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    best_acc = 0.0
    train_total_time = 0.0
    train_epoch_time = 0.0
    test_total_time = 0.0
    test_epoch_time = 0.0

    for epoch in range(60):

        # train
        train_running_loss = 0.0
        train_acc = 0.0
        net.train()
        train_start_time = time.time()
        for data in trainLoader:
            img_feature, label = data

            optimizer.zero_grad()
            img_feature = Variable(img_feature).cuda()
            label = Variable(label).long().cuda()
            #print(label)
            #print("label.shape", label.size())
            out = net(img_feature)
            #print("out.shape", out.size())
            pred = torch.max(out.data, 1)[1]
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
            train_acc += torch.sum(pred == label).item()
        scheduler.step()
        train_total_time += time.time() - train_start_time
        train_running_loss = train_running_loss / len(trainLoader)
        train_acc = train_acc / len(trainData)
        print("train: ", epoch)
        print("learning rate: ", scheduler.get_lr())
        print("train_running_loss: ", train_running_loss)
        print("train_acc: ", train_acc)

        # test
        test_running_loss = 0.0
        test_acc = 0.0
        test_start_time = time.time()

        for data in testLoader:
            optimizer.zero_grad()
            img_feature, label = data
            net.eval()
            optimizer.zero_grad()
            img_feature = Variable(img_feature).cuda()
            label = Variable(label).long().cuda()
            out = net(img_feature)
            pred = torch.max(out.data, 1)[1]
            loss = criterion(out, label)
            test_running_loss += loss.item()
            test_acc += torch.sum(pred == label).item()
        test_total_time += time.time() - test_start_time
        test_running_loss = test_running_loss / len(testLoader)
        test_acc = test_acc / len(testData)
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state_dict = copy.deepcopy(net.state_dict())
            best_epoch = epoch

        print("test_running_loss: ", test_running_loss)
        print("test_acc: ", test_acc)
        if train_acc == 1.0:
            train_epoch_time = train_total_time / (epoch + 1)
            test_epoch_time = test_total_time / (epoch + 1)
            break
        print("==================")

    if train_epoch_time == 0.0:
        train_epoch_time = train_total_time / 60
        test_epoch_time = test_total_time / 60
    print("train_epoch_time : ", train_epoch_time)
    print("test_epoch_time : ", test_epoch_time)
    state = {
        "epoch" : best_epoch,
        "model_state" : best_model_state_dict,
        "best_acc" : best_acc,
        "train_epoch_time" : train_epoch_time,
        "test_epoch_time" : test_epoch_time
    }
    total_best_acc.append(best_acc)
    total_train_epoch_time.append(train_epoch_time)
    total_test_epoch_time.append(test_epoch_time)
    # print(state)

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    save_path = "{}/{}_{}_{}_classroomClassification.pkl".format(model_dir, best_acc, args.backbone, exp_i)
    torch.save(state, save_path)
    fileName = args.backbone + ".csv"
    filePath = os.path.join(model_dir, fileName)
    if not os.path.exists(filePath):
        with open(filePath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "best_acc", "train_epoch_time", "test_epoch_time"])

    with open(filePath, "a+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([str(exp_i), str(best_acc), str(train_epoch_time), str(test_epoch_time)])

fileName = args.backbone + ".csv"
filePath = os.path.join(model_dir, fileName)
average_best_acc = sum(total_best_acc) / 5
average_train_epoch_time = sum(total_train_epoch_time) / 5
average_test_epoch_timie = sum(total_test_epoch_time) / 5
with open(filePath, "a+", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["average", str(average_best_acc), str(average_train_epoch_time), str(average_test_epoch_timie)])
