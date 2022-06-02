# #XuNet (with 6 groups).
# #"Structural Design of Convolutional Neural Networks for Steganalysis"
# #Date: 2020.09.21.
# #        1. According to Zeng Jishen's version, use slim.
# #        2. Correct the pre-processing high-pass filter with normalization.
# #Data: .mat.

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import numpy as np
import tensorflow.contrib.slim as slim
import os
import random
from scipy import ndimage
import scipy.io as sio
import h5py
import sys
import getopt
import time
import ConfigParser as configparser
# import configparser
sys.path.append('./tflib/')        # path for 'tflib' folder
from generator import *


BATCH_COVER = 32
BATCH_SIZE = BATCH_COVER*2
IMAGE_SIZE = 256
NUM_CHANNEL = 3
NUM_LABELS = 2
NUM_ITER = 0
MAX_ITER = 120000
NUM_SHOWTRAIN = 100  # show result eveary epoch
NUM_SHOWTEST = 5000
starter_learning_rate = 0.001

DIS_VER = 0
DB_SET = 'BOSS256_C'
COST_FUNC = 'HILL'
PAYLOAD = 0.4
NUM_SAMPLE = 7000
sample_file = 'boss256_color_trn.txt'

config_file = '/pubdata/qxh/BOSS256_C/config/trn_XuNet_HILL_BOSS_PPM256_p40.cfg'

opts, args = getopt.getopt(sys.argv[1:], 'hc:', ['help', 'config='])

for opt, val in opts:
    if opt == '-c':
        config_file = val
    if opt in ('-h', '--help'):
        print('Steganalyze samples by using XuNet (with 6 CNN groups).')
        print('  -c: the config file.')
        sys.exit()

is_train = True

# Read configs.
config_raw = configparser.RawConfigParser()
config_raw.read(config_file)
visible_devices = config_raw.get('Environment', 'CUDA_VISIBLE_DEVICES')
BATCH_COVER = config_raw.getint('Basic', 'BATCH_COVER')
BATCH_SIZE = 2*BATCH_COVER
# IMAGE_SIZE = 256
# NUM_CHANNEL = 1
# NUM_LABELS = 2
NUM_ITER = config_raw.getint('Basic', 'NUM_ITER')
MAX_ITER = config_raw.getint('Basic', 'MAX_ITER')
NUM_SHOWTRAIN = config_raw.getint('Basic', 'NUM_SHOWTRAIN')
NUM_SHOWTEST = config_raw.getint('Basic', 'NUM_SHOWTEST')

T = config_raw.getfloat('Basic', 'TEMPERATURE')

DIS_VER = config_raw.getint('Basic', 'DIS_VER')
DB_SET = config_raw.get('Basic', 'DB_SET')
COST_FUNC = config_raw.get('Basic', 'COST_FUNC')
PAYLOAD = config_raw.getfloat('Basic', 'PAYLOAD')
# BETA = config_raw.getfloat('Basic', 'BETA')
# GAMMA = config_raw.getfloat('Basic', 'GAMMA')
starter_learning_rate = config_raw.getfloat('Basic', 'starter_learning_rate')  # 0.001
decay_steps = config_raw.getfloat('Basic', 'decay_steps')  # 5000
decay_rate = config_raw.getfloat('Basic', 'decay_rate')  # 0.9

IS_DEBUG = config_raw.getfloat('Basic', 'IS_DEBUG')  # 0

model_dir = config_raw.get('Path', 'model_dir')
cover_dir = config_raw.get('Path', 'cover_dir')
sample_dir = config_raw.get('Path', 'sample_dir')
sample_file = config_raw.get('Path', 'sample_file')
log_dir = config_raw.get('Path', 'log_dir')
validation_cover = config_raw.get('Path', 'validation_cover')
validation_stego = config_raw.get('Path', 'validation_stego')
validation_file = config_raw.get('Path', 'validation_file')
validation_result = config_raw.get('Path', 'validation_result')

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices  # "7"
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(memory_gpu))
os.system('rm tmp')

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

if not os.path.exists(log_dir):
    os.mkdir(log_dir)

if not os.path.exists(validation_result):
    with open(validation_result, 'w+') as f_out:
        f_out.write('time:,%s\n' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        f_out.write('model:,%s\n' % (model_dir))
        f_out.write('cover:,%s\n' % (cover_dir))
        f_out.write('stego:,%s\n' % (sample_dir))
        f_out.write('sample: %s\n' % sample_file)
        f_out.write('iter,loss,Pfa,Pmd,PE,Dif\n')
else:
    with open(validation_result, 'a+') as f_out:
        f_out.write('time:,%s\n' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))


x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_LABELS])
is_train = tf.placeholder(tf.bool, name='is_train')

hpf = np.zeros([5, 5, NUM_CHANNEL, 1], dtype=np.float32)
# hpf[:, :, 0, 0] = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
#                             [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32)
for i in range(NUM_CHANNEL):
    hpf[:, :, i, 0] = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                                [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32)/12

kernel0 = tf.Variable(hpf, name="kernel0")
# conv0 = tf.nn.conv2d(x, kernel0, [1, 1, 1, 1], 'SAME', name="conv0")
conv0 = tf.nn.depthwise_conv2d(x, kernel0, [1, 1, 1, 1], 'SAME', name="conv0")


with tf.variable_scope("Group1") as scope:
    kernel1 = tf.Variable(tf.random_normal([5, 5, 3, 24], mean=0.0, stddev=0.01), name="kernel1")
    conv1 = tf.nn.conv2d(conv0, kernel1, [1, 1, 1, 1], padding='SAME', name="conv1")
    abs1 = tf.abs(conv1, name="abs1")

    bn1 = slim.layers.batch_norm(abs1, is_training=is_train, updates_collections=None, decay=0.05)
    tanh1 = tf.nn.tanh(bn1, name="tanh1")
    pool1 = tf.nn.avg_pool(tanh1, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool1")

with tf.variable_scope("Group2") as scope:
    kernel2_1 = tf.Variable(tf.random_normal([5, 5, 24, 48], mean=0.0, stddev=0.01), name="kernel2_1")
    conv2_1 = tf.nn.conv2d(pool1, kernel2_1, [1, 1, 1, 1], padding="SAME", name="conv2_1")

    bn2_1 = slim.layers.batch_norm(conv2_1, is_training=is_train, updates_collections=None, decay=0.05)
    tanh2_1 = tf.nn.tanh(bn2_1, name="tanh2_1")
    pool2 = tf.nn.avg_pool(tanh2_1, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool2_1")

with tf.variable_scope("Group3") as scope:
    kernel3 = tf.Variable(tf.random_normal([1, 1, 48, 96], mean=0.0, stddev=0.01), name="kernel3")
    conv3 = tf.nn.conv2d(pool2, kernel3, [1, 1, 1, 1], padding="SAME", name="conv3")

    bn3 = slim.layers.batch_norm(conv3, is_training=is_train, updates_collections=None, decay=0.05)

    relu3 = tf.nn.relu(bn3, name="relu3")
    pool3 = tf.nn.avg_pool(relu3, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool3")

with tf.variable_scope("Group4") as scope:
    kernel4_1 = tf.Variable(tf.random_normal([1, 1, 96, 192], mean=0.0, stddev=0.01), name="kernel4_1")
    conv4_1 = tf.nn.conv2d(pool3, kernel4_1, [1, 1, 1, 1], padding="SAME", name="conv4_1")

    bn4_1 = slim.layers.batch_norm(conv4_1, is_training=is_train, updates_collections=None, decay=0.05)
    relu4_1 = tf.nn.relu(bn4_1, name="relu4_1")
    pool4 = tf.nn.avg_pool(relu4_1, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool4")

with tf.variable_scope("Group5") as scope:
    kernel5 = tf.Variable(tf.random_normal([1, 1, 192, 384], mean=0.0, stddev=0.01), name="kernel5")
    conv5 = tf.nn.conv2d(pool4, kernel5, [1, 1, 1, 1], padding="SAME", name="conv5")

    bn5 = slim.layers.batch_norm(conv5, is_training=is_train, updates_collections=None, decay=0.05)
    relu5 = tf.nn.relu(bn5, name="relu5")
    pool5 = tf.nn.avg_pool(relu5, ksize=[1, 16, 16, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool5")

with tf.variable_scope('Group6') as scope:
    pool_shape = pool5.get_shape().as_list()
    pool_reshape = tf.reshape(pool5, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    weights = tf.Variable(tf.random_normal([384, 2], mean=0.0, stddev=0.01), name="weights")
    bias = tf.Variable(tf.random_normal([2], mean=0.0, stddev=0.01), name="bias")
    y_ = tf.matmul(pool_reshape, weights) + bias


vars = tf.trainable_variables()
params = [v for v in vars if (v.name.startswith('Group'))]

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('acc', accuracy)
correct_predictionCover, correct_predictionStego = tf.split(correct_prediction, 2, 0)
accuracyCover = tf.reduce_mean(tf.cast(correct_predictionCover, tf.float32))
accuracyStego = tf.reduce_mean(tf.cast(correct_predictionStego, tf.float32))
tf.summary.scalar('acc_cover', accuracyCover)
tf.summary.scalar('acc_stego', accuracyStego)

if T > 1:
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_/T))  # distill
else:
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_))

tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, trainable=False)
#starter_learning_rate = 0.001
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,5000, 0.9, staircase=True)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss, var_list=params, global_step=global_step)
#Passing global_step to minimize() will increment it at each step.

tf.summary.scalar('learn_rate', learning_rate)

data_x = np.zeros([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
data_y = np.zeros([BATCH_SIZE, NUM_LABELS])
# for i in range(0, BATCH_COVER):
#     data_y[i, 1] = 1
# for i in range(BATCH_COVER, BATCH_SIZE):
#     data_y[i, 0] = 1
data_y[0:BATCH_COVER, 1] = 1
data_y[BATCH_COVER:, 0] = 1

saver = tf.train.Saver(max_to_keep=10000)
merged = tf.summary.merge_all()
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    tf.global_variables_initializer().run()

    if NUM_ITER > 0:
        saver.restore(sess, model_dir+'/'+str(NUM_ITER)+'.ckpt')
        tf.assign(global_step, NUM_ITER)

    summary = tf.Summary()

    # fileList = []
    with open(sample_file, 'r') as f:
         lines=f.readlines()
         fileList = [a.strip() for a in lines]

    NUM_SAMPLE = len(fileList)

    # validationList = []
    with open(validation_file, 'r') as f:
        lines = f.readlines()
        validationList = [a.strip() for a in lines]

    NUM_VALIDATION = len(validationList)

    count = 0
    random.shuffle(fileList)
    for i in range(NUM_ITER, MAX_ITER+1):
        for j in range(BATCH_COVER):
            if count >= NUM_SAMPLE:
                count = 0
                #random.seed(i)
                random.shuffle(fileList)
            # dataC = sio.loadmat(sample_dir+'/'+fileList[count])
            # cover = dataC['coefC']
            # stego = dataC['coefS']
            cover, stego = read_ppm(cover_dir, sample_dir, fileList[count])
            data_x[j, :, :, :] = cover.astype(np.float32)
            data_x[j+BATCH_COVER, :, :, :] = stego.astype(np.float32)
            count = count+1
             
        _, tempL, tempA, tempC, tempS = sess.run([opt, loss, accuracy, accuracyCover, accuracyStego],
                                                 feed_dict={x: data_x, y: data_y, is_train: True})
       
        if i % NUM_SHOWTRAIN == 0:
            summary.ParseFromString(sess.run(merged, feed_dict={x: data_x, y: data_y, is_train: True}))
            writer.add_summary(summary, i)
            print('DB: %(d)s, Cost: %(c)s, payload: %(p).2f, T: %(t).2f batch results:' %
                  {'d': DB_SET, 'c': COST_FUNC, 'p': PAYLOAD, 't': T})
            print('iter=%(i)d, loss=%(s).4f, cover_acc=%(c).4f, stego_acc=%(t).4f, accuracy=%(a).4f...\n' %
                  {'i': i, 's': tempL, 'a': tempA, 'c': tempC, 't': tempS})

        if i % NUM_SHOWTEST == 0:
            # saver = tf.train.Saver()
            saver.save(sess, model_dir+'/'+str(i)+'.ckpt')
           
            # validation
            loss1 = np.array([])
            acc1 = np.array([])  # accuracy for validation set
            cover1 = np.array([])
            stego1 = np.array([])
            logit1 = []

            val_count = 0
            is_break = False
            random.shuffle(validationList)
            while val_count < NUM_VALIDATION:
                if is_break:  # the validation is maybe not entire mini-batches.
                    break

                for j in range(BATCH_COVER):
                    if val_count >= NUM_VALIDATION:
                        is_break = True
                        break

                    # dataC = sio.loadmat(validation_cover + '/' + validationList[val_count])
                    # cover = dataC['coefC']
                    # stego = dataC['coefS']
                    cover, stego = read_ppm(validation_cover, validation_stego, validationList[val_count])
                    data_x[j, :, :, :] = cover.astype(np.float32)
                    data_x[j + BATCH_COVER, :, :, :] = stego.astype(np.float32)
                    val_count = val_count + 1

                tempL, tempA, tempC, tempS, tempY = sess.run([loss, accuracy, accuracyCover, accuracyStego, y_],
                                                             feed_dict={x: data_x, y: data_y, is_train: False})
                loss1 = np.append(loss1, tempL)
                acc1 = np.append(acc1, tempA)
                cover1 = np.append(cover1, tempC)
                stego1 = np.append(stego1, tempS)
                logit1.append(tempY)

            loss_val = np.mean(loss1)
            acc_val = np.mean(acc1)
            acc_cover_val = np.mean(cover1)
            acc_stego_val = np.mean(stego1)
            summary.value.add(tag='val_loss', simple_value=loss_val)
            summary.value.add(tag='val_acc', simple_value=acc_val)
            summary.value.add(tag='val_acc_cover', simple_value=acc_cover_val)
            summary.value.add(tag='val_acc_stego', simple_value=acc_stego_val)
            writer.add_summary(summary, i)
            # print('time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time()))))
            print('Validation: iter = %(i)d, loss = %(s).4f, acc_cover = %(c).4f, acc_stego = %(t).4f, accuracy = %(a).4f...' %
                  {'i': i, 's': loss_val, 'a': acc_val, 'c': acc_cover_val, 't': acc_stego_val})
            print('--------------\n')
            ErrCover = 1-acc_cover_val
            ErrStego = 1-acc_stego_val
            ErrRate = 1-acc_val
            with open(validation_result, 'a+') as f_out:
                # f_out.write('iter,loss,Pfa,Pmd,PE,Dif\n')
                f_out.write('%d,%.4f,%.4f,%.4f,%.4f,%.4f\n' % (i, loss_val, ErrCover, ErrStego,
                                                               ErrRate, abs(ErrCover-ErrStego)/ErrRate))  # iteration,loss,cover error rate,stego error rate,error rate
            if IS_DEBUG == 1:
                loss_file = '%s/loss_%d.mat' % (log_dir, i)
                sio.savemat(loss_file, {'logits': logit1, 'loss': loss1, 'T': T})

