# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 09:15:08 2019

@author: kshan
"""

import tensorflow as tf

import random, os
import math
from datetime import datetime

from config import *
from make_train_data import *
from cnn_model import *

startTime = datetime.now()
print("Start_Time:",startTime)

print('Tensorflow version: {}'.format(tf.__version__))
print("Number of Classes: {}".format(flags.num_class))

if train_mode == 0:
    SAVE_PATH = SAVE_ROOT_PATH + 'model_train/'
else:
    SAVE_PATH = SAVE_ROOT_PATH + 'model_all/'

#model = tf.placeholder(tf.float32,[None,flags.num_class],name='model')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

y_pred = tf.nn.softmax(model, name='y_pred')
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#모델 학습
print('start')

    
train_batch = math.ceil(len(train_data)/batch_size)
test_batch = math.ceil(len(test_data)/batch_size)
print("Numbers of TRAINING Data: {}".format(len(train_data)))
print("TRAINING Step Numbers per Epoch: {}".format(train_batch))
print("Numbers of TEST Data: {}".format(len(test_data)))
print("TEST Step Numbers per Epoch: {}".format(test_batch))

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    
    sess.run(global_init)
    sess.run(local_init)

    if SET_INPUT_PIPELINE:
        sess.run(tr_iterator.initializer)
        if train_mode == 0:
            sess.run(te_iterator.initializer)
    else:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    lr = INIT_LEARNING_RATE
    check_step = INIT_CHECK_STEP
    
    cost_train = []
    check_epoch = []
    acc_train = []
    if train_mode == 0:
        cost_test = []
        acc_test = []
    
    startTime = datetime.now()
    
    for epoch in range(1,(train_epochs+1)):
        epochTime = datetime.now()
        train_cost_t = 0
        train_acc_t = 0
        if train_mode == 0:
            test_cost_t = 0
            test_acc_t = 0
        
        for ibatch in range(train_batch):
            if ibatch % 10 == 0:
                    print("Epoch[{}/{}]".format(epoch, train_epochs), 
                        "TRAIN Loop: {} / {}, \t".format(ibatch,train_batch), 
                        "Time: {}".format(datetime.now()))
            x_batch, y_batch = sess.run([tr_data, tr_label])
            x_batch = x_batch.reshape((-1, flags.height, flags.width, 1))
            if epoch == 1 and ibatch == 0:
                if SET_DISPLAY_IMG:
                    display_imgs(x_batch, ibatch)
            sess.run(optimizer, feed_dict={ X:x_batch, Y:y_batch, 
                                           keep_prob:train_keep_prob, learning_rate:lr})
            
        if epoch % check_step == 0:
            for ibatch in range(train_batch):
                if ibatch % 10 == 0:
                        print("Epoch[{}/{}]".format(epoch, train_epochs), 
                            "ESTIMATE Loop: {} / {}, \t".format(ibatch,train_batch), 
                            "Time: {}".format(datetime.now()))
                x_batch, y_batch = sess.run([tr_data, tr_label])
                x_batch = x_batch.reshape((-1, flags.height, flags.width, 1))
                cost_t, acc_t = sess.run([cost, accuracy], feed_dict={ X:x_batch, Y:y_batch, 
                                                                     keep_prob:test_keep_prob})
                train_cost_t += cost_t
                train_acc_t += acc_t
            train_cost = train_cost_t/train_batch
            train_acc = train_acc_t/train_batch
            cost_train.append(train_cost)
            acc_train.append(train_acc)
            check_epoch.append(epoch)
            
            if train_mode == 0:
                for jbatch in range(test_batch):
                    if jbatch % 10 == 0:
                            print("Epoch[{}/{}]".format(epoch, train_epochs), 
                                "Test Loop: {} / {}, \t".format(jbatch,test_batch), 
                                "Time: {}".format(datetime.now()))
                    x_batch, y_batch = sess.run([te_data, te_label])
                    x_batch = x_batch.reshape((-1, flags.height, flags.width, 1))
                    cost_t, acc_t = sess.run([cost, accuracy], feed_dict={ X:x_batch, Y:y_batch, 
                                                                         keep_prob:test_keep_prob})
                    test_cost_t += cost_t
                    test_acc_t += acc_t
                test_cost = test_cost_t/test_batch
                test_acc = test_acc_t/test_batch
                cost_test.append(test_cost)
                acc_test.append(test_acc)
                
            if train_mode == 0:
                    print("Train_Mode: {} / Epoch[{}/{}]...  ".format(train_mode, epoch, train_epochs))
                    print("\t Training Cost: {:.4f},\t".format(train_cost),"Test Cost: {:.4f}".format(test_cost))
                    print("\t Training Accuracy: {:.4f},\t".format(train_acc),"Test Accuracy: {:.4f}".format(test_acc))
            else:
                print("Train_Mode: {} / Epoch[{}/{}]...  ".format(train_mode, epoch, train_epochs))
                print("\t Training Cost: {:.4f},\t".format(train_cost))
                print("\t Training Accuracy: {:.4f}".format(train_acc))
                
        epoch_time = datetime.now()-epochTime
        print("Epoch[{}/{}] Time: {}".format(epoch, train_epochs, epoch_time))
        total_time = datetime.now()-startTime
        print("Total_Time: '{}'".format(total_time))
        print("Current_Time: '{}'".format(datetime.now()))
        print("\n")
            
        if epoch % check_step == 0:
            if epoch == 5:
                check_step = 5
            if epoch == 10:
                check_step = 10
            if epoch == 100:
                check_step = 50
            if epoch == 300:
                lr = lr * 0.1
            if epoch == 500:
                check_step = 100
                
                
            plt.figure()
            plt.plot(check_epoch, cost_train, 'bo--', label='Training Cost')
            if train_mode == 0:
                plt.plot(check_epoch, cost_test, 'r+--', label='Test Cost')
                plt.title('Training and Test Cost')
            else:
                plt.title('Training  Cost')
            plt.xlabel('Epochs ',fontsize=10)
            plt.ylabel('Cost',fontsize=10)
            plt.legend()
            plt.savefig(IMG_PATH+'Training_Test_cost_total[{}].png'.format(train_mode))
            plt.show()
            
            plt.figure()
            plt.plot(check_epoch, acc_train, 'bo--', label='Training Accuracy')
            if train_mode == 0:
                plt.plot(check_epoch,acc_test, 'r+--', label='Test Accuracy')
                plt.title('Training and Test Acuuracy')
            else:
                plt.title('Training Acuuracy')
            plt.xlabel('Epochs ',fontsize=10)
            plt.ylabel('Accuracy',fontsize=10)
            plt.legend()
            plt.savefig(IMG_PATH+'Training_Test_accuracy_total[{}].png'.format(train_mode))
            plt.show()
        if epoch % SAVE_STEP == 0:
            ### Save Model #########################
            saver = tf.train.Saver()
            saver.save(sess, SAVE_PATH+'char_recog_model_total[{}]'.format(train_mode), global_step=epoch)
            #######################################
        if epoch >= min_epochs:    
            if train_mode == 1:
                if train_acc >= target_train_acc:
                    break
           
            if train_mode == 0:
                if train_acc >= target_train_accuracy and test_acc >= target_test_accuracy:
                    break
        
    ### Save Final Model ###################
    saver = tf.train.Saver()
    saver.save(sess, SAVE_PATH+'char_recog_model_total[{}]'.format(train_mode))
    tf.train.write_graph(sess.graph.as_graph_def(), SAVE_PATH, 'char_recog_model_graph_total[{}].pb'.format(train_mode), as_text=False)
    ########################################
            
    print('학습완료!')
    
    
    
    #학습 결과 확인
#    x_test, y_test = train_data_batch(test_data,test_label,batch_size)
#    test_bx, test_by = sess.run([x_test, y_test])
#    is_correct = tf.equal(tf.argmax(model,1), tf.argmax(Y,1))
#    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
#    print('정확도:', sess.run(accuracy, feed_dict={X:test_bx, Y:test_by, keep_prob:1}))
    if not SET_INPUT_PIPELINE:
        coord.request_stop()
        coord.join(threads)


