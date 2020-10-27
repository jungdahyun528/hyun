# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 09:15:08 2019

@author: kshan
"""

import tensorflow as tf
import os
import shutil
import multiprocessing

tf.reset_default_graph()

#############################
# Make sub-folders for saving work results
#############################
shutil.rmtree('../graphs',ignore_errors=True)
os.makedirs('../graphs',exist_ok=True)
os.makedirs('../imgs',exist_ok=True)
os.makedirs('../log',exist_ok=True)
os.makedirs('../model',exist_ok=True)
os.makedirs('../model/model_train',exist_ok=True)
os.makedirs('../model/model_all',exist_ok=True)

SAVE_ROOT_PATH = "../model/"     # path for saving trained model

LOG_PATH = "../log/"        # path for writing log
IMG_PATH = "../imgs/"

DATA_PATH_PRT_GRAY = 'D:\DB_Survey\Eng_digits\DB\Char74K dataset\English\Fnt'
#DATA_PATH_PRT_GRAY = '../../../../01.접수/01.문자인식/data/DB_Survey/Eng_digits/DB/Char74K dataset/English/Fnt'

flags = tf.app.flags
flags.image_channel = 1   # 1: binary/grayscale 3:RGB 4:RGBA
#flags.image_color = 1   # 0: binary, 1: Gray
flags.width = 64
flags.height = 64

folder_list_prt_gray = os.listdir(DATA_PATH_PRT_GRAY)    # get the list of subfolder name in DATA_PATH
flags.num_class = len(folder_list_prt_gray)

initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)


handle = tf.placeholder(tf.string)

learning_rate = tf.placeholder(tf.float32)



#############################
# Training Config.
#############################
train_mode = 0 # 0: All-data Training, 1: Seperate Train-Test Data
train_epochs = 100
min_epochs = 30
batch_size = 512
INIT_LEARNING_RATE = 0.001
train_keep_prob = 0.5
test_keep_prob = 1.0
INIT_CHECK_STEP = 1
SAVE_STEP = 20
SET_INPUT_PIPELINE = True
#SET_INPUT_PIPELINE = False

target_train_accuracy = 0.999
target_test_accuracy = 0.97
#
#############################

#############################
# Cross Validation Config.
#############################
flags.num_fold = 10   # n-fold verification
test_size_rate = 0.2
SET_FOLD_EACH_CLASS = False
SET_SHUFFLE = True
SET_SHUFFLE_FIX = True
RANDOM_SEED = 4026
#############################

#############################
# Debugging Config.
#############################
SET_DEBUG_PRINT = True
#SET_DEBUG_PRINT = False
SET_DISPLAY_IMG = False
#############################
num_cores = multiprocessing.cpu_count()
#num_threads = num_cores
#num_cores = psutil.cpu_count(logical=True)
print("num_cores:{}".format(num_cores))