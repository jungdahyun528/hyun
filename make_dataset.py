# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 09:25:19 2019

@author: kshan
"""
import random, os
from config import *

fold_data = []
fold_label = []
fold_round = 0
test_data=[]
test_label=[]
for folder in folder_list_prt_gray:
#    DATA_PATH = os.path.join(DATA_ROOT_PATH, folder)
    DATA_PATH = DATA_PATH_PRT_GRAY +"/"+ folder + "/"
    filenames = os.listdir(DATA_PATH)
    file_list = []
    for filename in filenames:
        if filename.lower().endswith(".png"):
            filenamepath = os.path.join(DATA_PATH, filename)
            file_list.append(filenamepath)

    random.shuffle(file_list)

    label = folder_list_prt_gray.index(folder)
    test_size = int(len(file_list) * test_size_rate)
    test_data += file_list[:test_size]  # split test_data
    for i in range(test_size):
        test_label.append(label)  # stack test_label upto test_size

    filelist = file_list[test_size:]
 
    for data in filelist:

        fold_data.append(data)
        fold_label.append(label)
        fold_round += 1

def train_data_all(train_mode):
    train_data_i = []
    train_label_i = []
    for i in range(len(fold_data)):
        train_data_i.append(fold_data[i])
        train_label_i.append(fold_label[i])
    if train_mode != 0:
        train_data_i += test_data
        train_label_i += test_label
        
    train_data = []
    train_label = []
   
    randlist = random.sample(range(0,len(train_data_i)),len(train_data_i))
    for i in range(len(randlist)):
        train_data.append(train_data_i[randlist[i]])
        train_label.append(train_label_i[randlist[i]])

    return train_data, train_label, test_data, test_label