# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 09:27:10 2019

@author: kshan
"""

import tensorflow as tf

from datetime import datetime
import matplotlib.pyplot as plt
from config import *
from make_dataset import *
######################################
## Read Image Files and Queue Running
######################################
def get_train_queue(data,label, num_epochs=None):
    train_queue = tf.train.slice_input_producer([data,label],num_epochs=num_epochs,
                                            shuffle=True,seed=random.seed(datetime.now()))
    return train_queue

def read_train_data(train_queue):
    image_path = train_queue[0]
    label_org = train_queue[1]
    label = label_org
    image_contents = tf.read_file(image_path)
    image_org = tf.image.decode_png(image_contents,channels=flags.image_channel)
    image_cast = tf.cast(image_org, tf.float32)
    image_mul = tf.multiply(image_cast, 1.0/255.0)
    image_re = tf.image.resize_images(image_mul, [flags.height,flags.width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image_re,label

def train_data_batch(data,label):
    train_queue = get_train_queue(data,label)
    image,label = read_train_data(train_queue)
    image = tf.reshape(image,[flags.height*flags.width])
    batch_train_image,batch_train_label = tf.train.batch([image,label],batch_size=batch_size,allow_smaller_final_batch=True)
    batch_train_label_on_hot=tf.one_hot(batch_train_label, flags.num_class, on_value=1.0, off_value=0.0)
    return batch_train_image,batch_train_label_on_hot

######################################
## Read Image Files and Input Pipeline
######################################
def get_train_imgs(data,label):
    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    return dataset

def get_train_data(data,label):
    image = tf.read_file(data)
    image = tf.image.decode_png(image,channels=flags.image_channel)
    image = tf.cast(image,tf.float32)
    image = tf.multiply(image,1.0/255.0)
    image = tf.image.resize_images(image, [flags.height,flags.width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.reshape(image,[flags.height*flags.width,1])
    label = tf.cast(label, tf.int32)
    return image,label

def get_input_pipeline(data,label):
    dataset = get_train_imgs(data,label)
    dataset = dataset.shuffle(buffer_size=10*batch_size, reshuffle_each_iteration=True).repeat()
    dataset = dataset.map(get_train_data, num_parallel_calls=num_cores)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(num_cores)
    train_iterator = dataset.make_initializable_iterator()
    batch_train_image,batch_train_label = train_iterator.get_next()
    batch_train_label_on_hot=tf.one_hot(batch_train_label, flags.num_class, on_value=1.0, off_value=0.0)
    return batch_train_image,batch_train_label_on_hot,train_iterator

train_data, train_label, test_data, test_label = train_data_all(train_mode)

if SET_INPUT_PIPELINE:
    tr_data, tr_label, tr_iterator = get_input_pipeline(train_data,train_label)
    te_data, te_label, te_iterator = get_input_pipeline(test_data,test_label)
else:
    tr_data, tr_label = train_data_batch(train_data,train_label)
    te_data, te_label = train_data_batch(test_data,test_label)
    

#############################################
# Display Images to check
#############################################
def display_imgs(images, cnt):
    num_img = images.shape[0]
    
    row = int(num_img / 32)
    col = int(num_img / row)
    
    plt.figure(figsize=(col*2.4, row*2.4))

    for i in range(row):
        for j in range(col):
            plt.subplot(row, col, col*i+j+1)
            plt.imshow(images[col*i+j, ..., 0], cmap='gray')
    plt.savefig('../imgs/train_imgs-{}.png'.format(cnt))
    plt.show()
    return