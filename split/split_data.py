from asyncore import write
from cProfile import label
from operator import gt
import os
from random import random
from re import sub
import shutil
import glob
from random import uniform
from sysconfig import get_path


def split_data(src_root, partition_root):
    src_root = src_root
    train_l, train_u, val, test = [], [], [], []
    for shoe_classes in os.listdir(src_root):
        for img in os.listdir(os.path.join(src_root, shoe_classes)):
            a = uniform(0, 1)
            img_path = os.path.join(src_root, shoe_classes, img)
            if a > 0 and a <= 0.35:
                train_l.append(img_path)
            elif a > 0.35 and a <= 0.7:
                train_u.append(img_path)
            elif a > 0.7 and a <= 0.9:
                val.append(img_path)
            else:
                test.append(img_path)
    os.makedirs(partition_root, exist_ok=True)
    with open(os.path.join(partition_root, 'label.txt'), 'w') as f:
        [f.write(item+'\n') for item in train_l]
    with open(os.path.join(partition_root, 'unlabel.txt'), 'w') as f:
        [f.write(item+'\n') for item in train_u]
    with open(os.path.join(partition_root, 'val.txt'), 'w') as f:
        [f.write(item+'\n') for item in val]
    with open(os.path.join(partition_root, 'test.txt'), 'w') as f:
        [f.write(item+'\n') for item in test]


if __name__ == '__main__':
    src_root = '.' + os.sep + 'dataset' + os.sep + 'Shoe vs Sandal vs Boot Dataset'
    partition_root = '.' + os.sep + 'partitions'
