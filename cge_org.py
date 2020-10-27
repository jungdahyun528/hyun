import tensorflow as tf

import random, os
from datetime import datetime
print('start2')

DATA_PATH_PRT_GRAY = 'D:\DB_Survey\Eng_digits\DB\Char74K dataset\English\Fnt'
folder_list_prt_gray = os.listdir(DATA_PATH_PRT_GRAY)
test_size_rate = 0.2
img_height = img_width = 28
num_class = 36

fold_data = []
fold_label = []
fold_round = 0
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
    test_data=[]
    test_label=[]
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

    return train_data, train_label

def get_train_queue(data,label, num_epochs=None):
    train_queue = tf.train.slice_input_producer([data,label],num_epochs=num_epochs,
                                            shuffle=True,seed=random.seed(datetime.now()))
    return train_queue

def read_train_data(train_queue):
    image_path = train_queue[0]
    label_org = train_queue[1]
    label = label_org
    image_contents = tf.read_file(image_path)
    image_org = tf.image.decode_png(image_contents,channels=1)
    image_cast = tf.cast(image_org, tf.float32)
    image_mul = tf.multiply(image_cast, 1.0/255.0)
    image_re = tf.image.resize_images(image_mul, [28, 28],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image_re,label

def train_data_batch(data,label,batch_size):
    train_queue = get_train_queue(data,label)
    image,label = read_train_data(train_queue)
    image = tf.reshape(image,[img_height,img_width,1])
    batch_train_image,batch_train_label = tf.train.batch([image,label],batch_size=batch_size,allow_smaller_final_batch=True)
    batch_train_label_on_hot=tf.one_hot(batch_train_label, num_class, on_value=1.0, off_value=0.0)
    return batch_train_image,batch_train_label_on_hot


#CNN모델 구성
X=tf.placeholder(tf.float32, [None, 28,28,1])
Y=tf.placeholder(tf.float32, [None, 36])
keep_prob = tf.placeholder(tf.float32)

#1번째 CNN 계층
#16개 필터를 가진 4x4크기의 컨볼루션 계층을 정의
W1=tf.Variable(tf.random_normal([4,4,1,16], stddev=0.01))
#필터 슬라이딩 할 때 한 칸씩 슬라이딩을 수행
L1= tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)

#2x2크기의 폴링계층 정의
#2칸씩 슬라이딩
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#2번째 CNN 계층
#32개 필터를 가진 4X4크기의 컨볼루션 계층을 정의
W2 = tf.Variable(tf.random_normal([4,4,16,32], stddev=0.01))
L2 = tf.nn.conv2d(L1,W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#완전 연결계층(신경망)
W3 = tf.Variable(tf.random_normal([7*7*32,256], stddev=0.01))
#1차원 배열로 설정
L3 = tf.reshape(L2, [-1, 7*7*32])
#은닉층으로 256개의 뉴런을 연결
L3 = tf.matmul(L3,W3)
L3 = tf.nn.relu(L3)
#오버피팅을 막기 위해 드롭아웃 기법을 사용
L3 = tf.nn.dropout(L3, keep_prob)

#은닉층 256개의 뉴런을 입력으로 10갸의 분류를 만듭니다.
W4 = tf.Variable(tf.random_normal([256,36], stddev=0.01))
model = tf.matmul(L3, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


#모델 학습
print('start')
batch_size=512
data, label = train_data_all(0)
total_batch = int(len(data)/batch_size)

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
global_init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()

sess.run(global_init)
sess.run(local_init)

tr_data, tr_label = train_data_batch(data,label,batch_size)

    
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for epoch in range(3):
    total_cost=0
    
    for i in range(total_batch+1):
        x_batch, y_batch = sess.run([tr_data, tr_label])
        _, cost_val = sess.run([optimizer, cost], feed_dict={ X:x_batch, Y:y_batch, keep_prob:0.8})
        total_cost += cost_val
    print('반복:', '%04d'%(epoch+1), '평균 비용:', '{:.4f}'.format(total_cost/total_batch))
print('학습완료!')



#학습 결과 확인
x_test, y_test = train_data_batch(test_data,test_label,batch_size)
test_bx, test_by = sess.run([x_test, y_test])
is_correct = tf.equal(tf.argmax(model,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy, feed_dict={X:test_bx, Y:test_by, keep_prob:1}))
coord.request_stop()
coord.join(threads)


