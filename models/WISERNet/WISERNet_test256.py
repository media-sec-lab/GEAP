# #WISERENet
# #By: Qin Xinghong.
# #Date: 2020.10.27.
# #retrieve settings from the .csv file.
# #Test	Model	DIS_VER	Iteration	STE_VER	Stego_dir

import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim
import random
from scipy import ndimage
import scipy.io as sio
import sys
import getopt
# import ConfigParser as configparser
# # import configparser
sys.path.append('./tflib/')        # path for 'tflib' folder
from generator import *

HOST_NAME = '172.31.224.46'
BATCH_COVER = 16
BATCH_SIZE = 32
IMAGE_SIZE = 256
NUM_CHANNEL = 3
NUM_LABELS = 2
NUM_ITER = 200000
COST_FUNC = 'SUNIWARD'
PAYLOAD = 0.4
record_stego = 0
# NUM_SHOWTRAIN = 100 #show result eveary epoch
# NUM_SHOWTEST = 20000
N_MAGNIFICATION = 9

sample_file = '/pubdata/qxh/BOSS256_C/config_color/boss256_color_tst.txt'
cover_dir = '/pubdata/qxh/BOSS256_C/BOSS_PPM256'
result_file = ''

config_file = './config_color/tst_YeNet_HILL_BOSS256_PPM_p40.csv'
opts, args = getopt.getopt(sys.argv[1:], 'hc:t:', ['help', 'config=', 'host='])

for opt, val in opts:
    if opt == '-c':
        config_file = val
    if opt == '-t':
        HOST_NAME = val
    if opt in ('-h', '--help'):
        print('Steganalyze samples by using YuNet.')
        print('  -c: the config file.')
        sys.exit()

# Read configs.
dis_ver_list = np.array([], dtype=np.int)
dis_model_list = np.array([], dtype=np.str)
dis_iter_list = np.array([], dtype=np.int)
ste_ver_list = np.array([], dtype=np.int)
ste_path_list = np.array([], dtype=np.str)
with open(config_file, 'r') as f_id:
    lines = f_id.readlines()
    for s_line in lines:
        s_temp = s_line.split(',')
        key_t = s_temp[0]
        key_t = key_t.upper()
        if key_t == 'BATCH_COVER':
            BATCH_COVER = int(s_temp[1])
            BATCH_SIZE = 2*BATCH_COVER
        elif key_t == 'COST_FUNC':
            COST_FUNC = s_temp[1]
        elif key_t == 'IS_SCA':
            IS_SCA = int(s_temp[1])
        elif key_t == 'PAYLOAD':
            PAYLOAD = float(s_temp[1])
        elif key_t == 'DB_SET':
            DB_SET = s_temp[1]
        elif key_t == 'COVER_DIR':
            if len(s_temp) > 2 and len(s_temp[2]) > 3 and s_temp[2].strip() != HOST_NAME:
                continue
            cover_dir = s_temp[1]
        elif key_t == 'SAMPLE_FILE':
            if len(s_temp) > 2 and len(s_temp[2]) > 3 and s_temp[2].strip() != HOST_NAME:
                continue
            sample_file = s_temp[1]
        elif key_t == 'RECORD_STEGO':
            record_stego = int(s_temp[1])
        elif key_t == 'RESULT_FILE':
            if len(s_temp) > 2 and len(s_temp[2]) > 3 and s_temp[2].strip() != HOST_NAME:
                continue
            result_file = s_temp[1]
        elif key_t == 'RECORD_DIR':
            if len(s_temp) > 2 and len(s_temp[2]) > 3 and s_temp[2].strip() != HOST_NAME:
                continue
            record_dir = s_temp[1]
        elif key_t == 'TEST':
            if len(s_temp) > 6 and len(s_temp[6]) > 3 and s_temp[6].strip() != HOST_NAME:
                continue
            dis_model_list = np.append(dis_model_list, s_temp[1])
            dis_ver_list = np.append(dis_ver_list, int(s_temp[2]))
            dis_iter_list = np.append(dis_iter_list, int(s_temp[3]))
            ste_ver_list = np.append(ste_ver_list, int(s_temp[4]))
            ste_path_list = np.append(ste_path_list, s_temp[5])

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices  # 5
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(memory_gpu))
os.system('rm tmp')

fileList = []
if len(sample_file) > 3:
    with open(sample_file, 'r') as f:
        lines = f.readlines()
        fileList = fileList + [a.strip() for a in lines]
else:
    for (dirpath, dirnames, filenames) in os.walk(cover_dir):
        fileList = filenames
NUM_SAMPLE = len(fileList)

# np.set_printoptions(threshold='nan')

x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_LABELS])
is_train = tf.placeholder(tf.bool, name='is_train')

hpf = np.zeros([5, 5, 3, 30], dtype=np.float32)  # [height,width,input,output]

for idx_k in range(NUM_CHANNEL):
    hpf[1:4, 1:4, idx_k, 0] = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=np.float32)
    hpf[1:4, 1:4, idx_k, 1] = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32)
    hpf[1:4, 1:4, idx_k, 2] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32)
    hpf[1:4, 1:4, idx_k, 3] = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32)
    hpf[1:4, 1:4, idx_k, 4] = np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]], dtype=np.float32)
    hpf[1:4, 1:4, idx_k, 5] = np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]], dtype=np.float32)
    hpf[1:4, 1:4, idx_k, 6] = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32)
    hpf[1:4, 1:4, idx_k, 7] = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)

    hpf[1:4, 1:4, idx_k, 8] = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32)
    hpf[1:4, 1:4, idx_k, 9] = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32)
    hpf[1:4, 1:4, idx_k, 10] = np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]], dtype=np.float32)
    hpf[1:4, 1:4, idx_k, 11] = np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]], dtype=np.float32)

    hpf[:, :, idx_k, 12] = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -3, 3, -1],
                                 [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32)
    hpf[:, :, idx_k, 13] = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -3, 0, 0],
                                 [0, 0, 0, 3, 0], [0, 0, 0, 0, -1]], dtype=np.float32)
    hpf[:, :, idx_k, 14] = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -3, 0, 0],
                                 [0, 0, 3, 0, 0], [0, 0, -1, 0, 0]], dtype=np.float32)
    hpf[:, :, idx_k, 15] = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, -3, 0, 0],
                                 [0, 3, 0, 0, 0], [-1, 0, 0, 0, 0]], dtype=np.float32)
    hpf[:, :, idx_k, 16] = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 3, -3, 1, 0],
                                 [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32)
    hpf[:, :, idx_k, 17] = np.array([[-1, 0, 0, 0, 0], [0, 3, 0, 0, 0], [0, 0, -3, 0, 0],
                                 [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=np.float32)
    hpf[:, :, idx_k, 18] = np.array([[0, 0, -1, 0, 0], [0, 0, 3, 0, 0], [0, 0, -3, 0, 0],
                                 [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32)
    hpf[:, :, idx_k, 19] = np.array([[0, 0, 0, 0, -1], [0, 0, 0, 3, 0], [0, 0, -3, 0, 0],
                                 [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32)

    hpf[1:4, 1:4, idx_k, 20] = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32)

    hpf[:, :, idx_k, 21] = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                                 [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32)

    hpf[1:4, 1:4, idx_k, 22] = np.array([[-1, 2, -1], [2, -4, 2], [0, 0, 0]], dtype=np.float32)
    hpf[1:4, 1:4, idx_k, 23] = np.array([[-1, 2, 0], [2, -4, 0], [-1, 2, 0]], dtype=np.float32)
    hpf[1:4, 1:4, idx_k, 24] = np.array([[0, 0, 0], [2, -4, 2], [-1, 2, -1]], dtype=np.float32)
    hpf[1:4, 1:4, idx_k, 25] = np.array([[0, 2, -1], [0, -4, 2], [0, 2, -1]], dtype=np.float32)

    hpf[:, :, idx_k, 26] = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                                 [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32)
    hpf[:, :, idx_k, 27] = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-2, 8, -12, 8, -2],
                                 [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32)
    hpf[:, :, idx_k, 28] = np.array([[0, 0, -2, 2, -1], [0, 0, 8, -6, 2], [0, 0, -12, 8, -2],
                                 [0, 0, 8, -6, 2], [0, 0, -2, 2, -1]], dtype=np.float32)
    hpf[:, :, idx_k, 29] = np.array([[-1, 2, -2, 0, 0], [2, -6, 8, 0, 0], [-2, 8, -12, 0, 0],
                                 [2, -6, 8, 0, 0], [-1, 2, -2, 0, 0]], dtype=np.float32)

OUT_D2 = 8*N_MAGNIFICATION
OUT_D3 = 32*N_MAGNIFICATION
OUT_D4 = 128*N_MAGNIFICATION

DECAY_RATE = 0.95  # 0.95  # 0.05
STDDEV = 0.1

with tf.variable_scope("Group1") as scope:

    kernel1 = tf.Variable(hpf, name="kernel1")  # [height,width,input,output]
    conv1 = tf.nn.depthwise_conv2d(x, kernel1, [1, 1, 1, 1], padding='SAME', name="conv1")

with tf.variable_scope("Group2") as scope:
    # kernel2 = tf.get_variable(shape=[5, 5, 90, OUT_D2], initializer=tf.contrib.layers.xavier_initializer(), name="kernel2")
    # conv2_1 = tf.nn.conv2d(conv1, kernel2, [1, 2, 2, 1], padding="SAME", name="conv2_1")
    conv2_1 = tf.layers.conv2d(conv1, filters=OUT_D2, kernel_size=5, padding='SAME')
    conv2_2 = tf.abs(conv2_1, name="conv2_2")

    bn2 = slim.layers.batch_norm(conv2_2, scale=True, is_training=is_train, updates_collections=None, decay=DECAY_RATE)

    relu2 = tf.nn.relu(bn2, name="relu2")

    pool2 = tf.nn.avg_pool(relu2, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")

with tf.variable_scope("Group3") as scope:
    kernel3 = tf.get_variable(shape=[3, 3, OUT_D2, OUT_D3], initializer=tf.contrib.layers.xavier_initializer(), name="kernel3")
    conv3 = tf.nn.conv2d(pool2, kernel3, [1, 1, 1, 1], padding="SAME", name="conv3")

    bn3 = slim.layers.batch_norm(conv3, scale=True, is_training=is_train, updates_collections=None, decay=DECAY_RATE)

    relu3 = tf.nn.relu(bn3, name="relu3")

    pool3 = tf.nn.avg_pool(relu3, ksize=[1, 5, 5, 1], strides=[1, 4, 4, 1], padding="SAME", name="pool3")

with tf.variable_scope("Group4") as scope:
    kernel4 = tf.get_variable(shape=[3, 3, OUT_D3, OUT_D4], initializer=tf.contrib.layers.xavier_initializer(), name="kernel4")
    conv4 = tf.nn.conv2d(pool3, kernel4, [1, 1, 1, 1], padding="SAME", name="conv4")

    bn4 = slim.layers.batch_norm(conv4, scale=True, is_training=is_train, updates_collections=None, decay=DECAY_RATE)

    relu4 = tf.nn.relu(bn4, name="relu4")
    pool4 = tf.nn.avg_pool(relu4, ksize=[1, 32, 32, 1], strides=[1, 32, 32, 1], padding="SAME", name="pool4")

    pool_shape4 = pool4.get_shape().as_list()
    feat_len = pool_shape4[1] * pool_shape4[2] * pool_shape4[3]
    pool_reshape4 = tf.reshape(pool4, [pool_shape4[0], feat_len])

with tf.variable_scope('Group5') as scope:
    weights5 = tf.Variable(tf.random_normal([feat_len, 800], mean=0.0, stddev=STDDEV), name="weights5")
    # tf.add_to_collection('losses', slim.l2_regularizer(0.01)(weights5))
    bias5 = tf.Variable(tf.zeros([800]), name="bias5")
    feat5 = tf.matmul(pool_reshape4, weights5) + bias5
    relu5 = tf.nn.relu(feat5, name="relu5")

with tf.variable_scope('Group6') as scope:
    weights6 = tf.Variable(tf.random_normal([800, 400], mean=0.0, stddev=STDDEV), name="weights6")
    # tf.add_to_collection('losses', slim.l2_regularizer(0.01)(weights6))
    bias6 = tf.Variable(tf.zeros([400]), name="bias6")
    feat6 = tf.matmul(relu5, weights6) + bias6
    relu6 = tf.nn.relu(feat6, name="relu6")

with tf.variable_scope('Group7') as scope:
    weights7 = tf.Variable(tf.random_normal([400, 200], mean=0.0, stddev=STDDEV), name="weights7")
    # tf.add_to_collection('losses', slim.l2_regularizer(0.01)(weights7))
    bias7 = tf.Variable(tf.zeros([200]), name="bias7")
    feat7 = tf.matmul(relu6, weights7) + bias7
    relu7 = tf.nn.relu(feat7, name="relu7")

with tf.variable_scope('Group8') as scope:
    weights8 = tf.Variable(tf.random_normal([200, 2], mean=0.0, stddev=STDDEV), name="weights8")
    # tf.add_to_collection('losses', slim.l2_regularizer(0.01)(weights8))
    bias8 = tf.Variable(tf.zeros([2]), name="bias8")
    y_ = tf.matmul(relu7, weights8) + bias8


vars = tf.trainable_variables()
params = [v for v in vars if (v.name.startswith('Group'))]

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
correct_predictionCover, correct_predictionStego = tf.split(correct_prediction, 2, 0)
accuracyCover = tf.reduce_mean(tf.cast(correct_predictionCover, tf.float32))
accuracyStego = tf.reduce_mean(tf.cast(correct_predictionStego, tf.float32))

loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_)
loss = tf.reduce_mean(loss_)

data_x = np.zeros([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
data_y = np.zeros([BATCH_SIZE, NUM_LABELS])

for i in range(0, BATCH_COVER):
    data_y[i, 1] = 1
for i in range(BATCH_COVER, BATCH_SIZE):
    data_y[i, 0] = 1

for DIS_VER, model_dir, NUM_ITER, STEGO_VER, sample_dir \
        in zip(dis_ver_list, dis_model_list, dis_iter_list, ste_ver_list, ste_path_list):
    model_dir = model_dir.strip()
    sample_dir = sample_dir.strip()
    print('dis_ver: %d, model: %s\nsample_ver: %d, sample_dir: %s\n' % (DIS_VER, model_dir, STEGO_VER, sample_dir))
    with tf.Session() as sess:
        # for epoch in range(0,100000000,10000):
        # tf.initialize_all_variables().run()
        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        saver.restore(sess, model_dir+'/'+str(NUM_ITER)+'.ckpt')

        resultCover = np.array([])  # accuracy for training set
        resultStego = np.array([])  # accuracy for training set

        Loss = np.array([])
        Accuracy = np.array([])
        AccuracyCover = np.array([])
        AccuracyStego = np.array([])
        ArrayCount = [0] * BATCH_COVER

        result = np.array([]) #accuracy for training set
        count = 0
        while count < NUM_SAMPLE:
            for j in range(BATCH_COVER):
                if count >= NUM_SAMPLE:
                    break

                ArrayCount[j] = count
                cover, stego = read_ppm(cover_dir, sample_dir, fileList[count])
                data_x[j, :, :, :] = cover.astype(np.float32)
                data_x[j + BATCH_COVER, :, :, :] = stego.astype(np.float32)
                count = count+1

            # print()
            tempA, tempB, tempC, temp0, temp1 = sess.run([accuracy, accuracyCover, accuracyStego, loss, loss_],
                                                         feed_dict={x: data_x, y: data_y, is_train: False})
            print('ver: %d-%d, count: %d, loss: %.4f, cover accu: %.4f, stego accu: %.4f, accu: %.4f' %
                  (DIS_VER, STEGO_VER, count, np.mean(temp0), np.mean(tempB), np.mean(tempC), np.mean(tempA)))
            Accuracy = np.insert(Accuracy, 0, tempA)
            AccuracyCover = np.insert(AccuracyCover, 0, tempB)
            AccuracyStego = np.insert(AccuracyStego, 0, tempC)
            Loss = np.insert(Loss, 0, temp0)
            # record loss
            if record_stego == 1:
                for j in range(BATCH_COVER):
                    stego_name = fileList[ArrayCount[j]]
                    f_name, f_ext = os.path.splitext(stego_name)
                    f_single = record_dir + '/' + f_name + '.csv'
                    if os.path.isfile(f_single):
                        with open(f_single, 'a+') as f_s:
                            f_s.write('%d,%d,%d,%s,%.4f,%s,%.2f,%s,%s\n' %
                                      (DIS_VER, NUM_ITER, STEGO_VER, model_dir, temp1[j + BATCH_COVER],
                                       COST_FUNC, PAYLOAD, sample_dir, stego_name))
                    else:
                        with open(f_single, 'w+') as f_s:
                            f_s.write('Discriminator,Iteration,Stego ver,model,Loss,Cost,Payload,path,sample\n')
                            f_s.write('%d,%d,%d,%s,%.4f,%s,%.2f,%s,%s\n' %
                                      (DIS_VER, NUM_ITER, STEGO_VER, model_dir, temp1[j + BATCH_COVER],
                                       COST_FUNC, PAYLOAD, sample_dir, stego_name))

        ErrCover = 1 - np.mean(AccuracyCover)
        ErrStego = 1 - np.mean(AccuracyStego)
        ErrRate = 1 - np.mean(Accuracy)
        Loss_i = np.mean(Loss)
        print('iter: %d, loss: %.4f, err cover: %.4f, err stego: %.4f, err rate: %.4f\n' %
              (NUM_ITER, Loss_i, ErrCover, ErrStego, ErrRate))

        if len(result_file) > 3:
            if not os.path.isfile(result_file):
                with open(result_file, 'w+') as f_out:
                    f_out.write(
                        'discriminator,iter,cost,stego ver,payload,loss,cover error rate,stego error rate,error rate,sample list,model,sample dir\n')
                    f_out.write('%d,%d,%s,%d,%.2f,%.4f,%.4f,%.4f,%.4f,%s,%s,%s\n' %
                                (DIS_VER, NUM_ITER, COST_FUNC, STEGO_VER, PAYLOAD, Loss_i,
                                 ErrCover, ErrStego, ErrRate, sample_file, model_dir, sample_dir))
            else:
                with open(result_file, 'a+') as f_out:
                    # f_out.write('discriminator,iter,cost,stego ver,payload,loss,cover error rate,stego error rate,error rate,sample list,model,sample dir\n')
                    f_out.write('%d,%d,%s,%d,%.2f,%.4f,%.4f,%.4f,%.4f,%s,%s,%s\n' %
                                (DIS_VER, NUM_ITER, COST_FUNC, STEGO_VER, PAYLOAD, Loss_i,
                                 ErrCover, ErrStego, ErrRate, sample_file, model_dir, sample_dir))

