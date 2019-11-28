# -*- coding:utf-8 -*-
# Author : xuan
# Date : 2019/11/3 17:31
import numpy as np
from PIL import Image
import os
from torchvision import transforms as tfs
from sklearn.model_selection import StratifiedShuffleSplit

segmentation_root_data_path = r"D:\dataset\classroom_6\ActionsClassifySegmentation"
not_segmentation_root_data_path = r"D:\dataset\classroom_6\ActionsClassifyResizeRename"

labels = os.listdir(segmentation_root_data_path)
im_tfs = tfs.Compose([
        tfs.Resize((299, 299)),
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

segmentation_data = []
not_segmentation_data = []
label = []
for i in labels:
    segmentation_data_path = os.path.join(segmentation_root_data_path, i)
    not_segmentation_data_path = os.path.join(not_segmentation_root_data_path, i)
    for fileName in os.listdir(segmentation_data_path):
        segmenatationFilePath = os.path.join(segmentation_data_path, fileName)
        segmentationImg = Image.open(segmenatationFilePath)
        segmentationImg = im_tfs(segmentationImg)
        segmentation_data.append(segmentationImg.numpy())

        notSegmenatationFilePath = os.path.join(not_segmentation_data_path, fileName)
        notSegmentationImg = Image.open(notSegmenatationFilePath)
        notSegmentationImg = im_tfs(notSegmentationImg)
        not_segmentation_data.append(notSegmentationImg.numpy())

        label.append(int(i)-1)
        print(i + " : " + fileName)

segmentation_data = np.array(segmentation_data)
not_segmentation_data = np.array(not_segmentation_data)
label = np.array(label)

ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

train_idx, test_idx = next(ss.split(segmentation_data, label))

segmentation_train_x, segmentation_train_y = segmentation_data[train_idx], label[train_idx]
segmentation_test_x, segmentation_test_y = segmentation_data[test_idx], label[test_idx]

np.savez("segmentation_data_split.npz", train_x=segmentation_train_x, train_y=segmentation_train_y, test_x=segmentation_test_x, test_y=segmentation_test_y)

not_segmentation_train_x, not_segmentation_train_y = not_segmentation_data[train_idx], label[train_idx]
not_segmentation_test_x, not_segmentation_test_y = not_segmentation_data[test_idx], label[test_idx]

np.savez("not_segmentation_data_split.npz", train_x=not_segmentation_train_x, train_y=not_segmentation_train_y, test_x=not_segmentation_test_x, test_y=not_segmentation_test_y)


