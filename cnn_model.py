#4개의 컨볼루션 계층, 2개의 완전연결 계층, dropout, final layer
#input image=96*96
#https://bcho.tistory.com/1178?category=555440
import tensorflow as tf

import random, os
from datetime import datetime

from config import *

def max_pool(input_data, name):
    return tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def conv(input_data,pre_layer_size, now_layer_size, name):
    flags.conv_filter_size = 3
    flags.stride1 = 1
    
    with tf.name_scope(name):
        W_conv = tf.Variable(tf.truncated_normal(
                        [flags.conv_filter_size,flags.conv_filter_size, pre_layer_size, now_layer_size],
                                              stddev=0.1))
        b = tf.Variable(tf.truncated_normal(
                        [now_layer_size],stddev=0.1))
        h_conv = tf.nn.conv2d(input_data,W_conv,strides=[1,1,1,1],padding='SAME')
        h_conv_relu = tf.nn.relu(tf.add(h_conv,b))
        return h_conv_relu


# fully connected layer 1
def fc1(input_data, pre_layer_size):
    input_layer_size = 8*8*pre_layer_size
    flags.fc1_layer_size = 512
    
    with tf.name_scope('fc_1'):
        # 앞에서 입력받은 다차원 텐서를 fcc에 넣기 위해서 1차원으로 피는 작업
        input_data_reshape = tf.reshape(input_data, [-1, input_layer_size])
        W_fc1 = tf.Variable(tf.truncated_normal([input_layer_size,flags.fc1_layer_size],stddev=0.1))
        b_fc1 = tf.Variable(tf.truncated_normal(
                        [flags.fc1_layer_size],stddev=0.1))
        h_fc1 = tf.add(tf.matmul(input_data_reshape,W_fc1) , b_fc1) # h_fc1 = input_data*W_fc1 + b_fc1
        h_fc1_relu = tf.nn.relu(h_fc1)
    
    return h_fc1_relu
    
# fully connected layer 2
def fc2(input_data):
    flags.fc2_layer_size = 256
    
    with tf.name_scope('fc_2'):
        W_fc2 = tf.Variable(tf.truncated_normal([flags.fc1_layer_size,flags.fc2_layer_size],stddev=0.1))
        b_fc2 = tf.Variable(tf.truncated_normal(
                        [flags.fc2_layer_size],stddev=0.1))
        h_fc2 = tf.add(tf.matmul(input_data,W_fc2) , b_fc2) # h_fc2 = input_data*W_fc2 + b_fc2
        h_fc2_relu = tf.nn.relu(h_fc2)
    
    return h_fc2_relu

# final layer
def final_out(input_data):

    with tf.name_scope('final_out'):
        W_fo = tf.Variable(tf.truncated_normal([flags.fc2_layer_size,flags.num_class],stddev=0.1))
        b_fo = tf.Variable(tf.truncated_normal(
                        [flags.num_class],stddev=0.1))
        h_fo = tf.add(tf.matmul(input_data,W_fo) , b_fo) # h_fc1 = input_data*W_fc1 + b_fc1
        
    # 최종 레이어에 softmax 함수는 적용하지 않았다.
    return h_fo

X=tf.placeholder(tf.float32, [None, flags.height, flags.width,1], name='X')
Y=tf.placeholder(tf.float32, [None, flags.num_class], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

flags.conv1_layer_size = 16
conv1_1 = conv(X,1,flags.conv1_layer_size, "conv1_1") #[3,3,1,16]
conv1_2 = conv(conv1_1,flags.conv1_layer_size,flags.conv1_layer_size, "conv1_2") #[3,3,16,16]
pool1 = max_pool(conv1_2, 'pool1')

flags.conv2_layer_size = 32
conv2_1 = conv(pool1,flags.conv1_layer_size,flags.conv2_layer_size, "conv2_1") #[3,3,16,32]
conv2_2 = conv(conv2_1,flags.conv2_layer_size,flags.conv2_layer_size, "conv2_2") #[3,3,32,32]
pool2 = max_pool(conv2_2, 'pool2') 

flags.conv3_layer_size = 64
conv3_1 = conv(pool2, flags.conv2_layer_size, flags.conv3_layer_size, "conv3_1") #[3,3,32,64]
conv3_2 = conv(conv3_1, flags.conv3_layer_size, flags.conv3_layer_size, "conv3_2") #[3,3,64,64]
pool3 = max_pool(conv3_2, 'pool3')

fc1 = fc1(pool3, flags.conv3_layer_size) #[8*8*64,512]
dropout1 = tf.nn.dropout(fc1,keep_prob) 

fc2 = fc2(dropout1) #[512,256]
dropout2 = tf.nn.dropout(fc2,keep_prob)

model = final_out(dropout2) #[256,36]