# # WISERNet: Wider Separate-Then-Reunion Network for Steganalysis of Color Images
# # the config file contains sample dir, model dir, eta.

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os
import random
import sys
import time
import getopt
import ConfigParser as configparser
# import configparser
sys.path.append('./tflib/')        # path for 'tflib' folder
from generator import *

BATCH_COVER = 8
BATCH_SIZE = 2*BATCH_COVER
IMAGE_SIZE = 256
NUM_CHANNEL = 3
NUM_LABELS = 2
NUM_ITER = 0
MAX_ITER = 300000
NUM_SHOWTRAIN = 100  # show result eveary epoch
NUM_SHOWTEST = 10000

N_MAGNIFICATION = 9

model_dir = ''
cover_dir = ''
stego_dir = ''
validation_cover = ''
validation_stego = ''
validation_proc = ''
sample_file = 'boss256_color_trn.txt'
config_file = '/data1/dataset/BOSS256_C/config/trn_WISERENet_CPV_BOSS256_PPM_p40.cfg'

opts, args = getopt.getopt(sys.argv[1:], 'hc:', ['help', 'config='])

for opt, val in opts:
    if opt == '-c':
        config_file = val
    if opt in ('-h', '--help'):
        print('Train YuNet.')
        print('  -c: the config file.')
        sys.exit()

# Read configs.
config_raw = configparser.RawConfigParser()
config_raw.read(config_file)
BATCH_COVER = config_raw.getint('Basic', 'BATCH_COVER')
BATCH_SIZE = 2*BATCH_COVER
IMAGE_SIZE = config_raw.getint('Basic', 'IMAGE_SIZE')  # 256
NUM_ITER = config_raw.getint('Basic', 'NUM_ITER')
MAX_ITER = config_raw.getint('Basic', 'MAX_ITER')
NUM_SHOWTRAIN = config_raw.getint('Basic', 'NUM_SHOWTRAIN')
NUM_SHOWTEST = config_raw.getint('Basic', 'NUM_SHOWTEST')

DIS_VER = config_raw.getint('Basic', 'DIS_VER')
DB_SET = config_raw.get('Basic', 'DB_SET')
COST_FUNC = config_raw.get('Basic', 'COST_FUNC')
PAYLOAD = config_raw.getfloat('Basic', 'PAYLOAD')

starter_learning_rate = config_raw.getfloat('Basic', 'starter_learning_rate')  # 0.001
decay_steps = config_raw.getfloat('Basic', 'decay_steps')  # 5000
decay_rate = config_raw.getfloat('Basic', 'decay_rate')  # 0.75

model_dir = config_raw.get('Path', 'model_dir')
cover_dir = config_raw.get('Path', 'cover_dir')
stego_dir = config_raw.get('Path', 'stego_dir')
sample_file = config_raw.get('Path', 'sample_file')
log_dir = config_raw.get('Path', 'log_dir')
validation_cover = config_raw.get('Path', 'validation_cover')
validation_stego = config_raw.get('Path', 'validation_stego')
validation_file = config_raw.get('Path', 'validation_file')
validation_result = config_raw.get('Path', 'validation_result')

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
        f_out.write('stego:,%s\n' % (stego_dir))
        f_out.write('iter,loss,Pfa,Pmd,PE,Dif\n')
else:
    with open(validation_result, 'a+') as f_out:
        f_out.write('time:,%s\n' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

fileList = []
if len(sample_file) > 0:
    with open(sample_file, 'r') as f:
        lines = f.readlines()
        fileList = fileList + [a.strip() for a in lines]
else:
    for (dirpath, dirnames, filenames) in os.walk(stego_dir):
        fileList = filenames
NUM_SAMPLE = len(fileList)

validationList = []
with open(validation_file, 'r') as f:
    lines = f.readlines()
    validationList = [a.strip() for a in lines]

NUM_VALIDATION = len(validationList)

print('Training samples: %d, validation samples: %d...\n' % (NUM_SAMPLE, NUM_VALIDATION))

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
    bias6 = tf.Variable(tf.zeros([400]), name="bias6")
    feat6 = tf.matmul(relu5, weights6) + bias6
    relu6 = tf.nn.relu(feat6, name="relu6")

with tf.variable_scope('Group7') as scope:
    weights7 = tf.Variable(tf.random_normal([400, 200], mean=0.0, stddev=STDDEV), name="weights7")
    bias7 = tf.Variable(tf.zeros([200]), name="bias7")
    feat7 = tf.matmul(relu6, weights7) + bias7
    relu7 = tf.nn.relu(feat7, name="relu7")

with tf.variable_scope('Group8') as scope:
    weights8 = tf.Variable(tf.random_normal([200, 2], mean=0.0, stddev=STDDEV), name="weights8")
    bias8 = tf.Variable(tf.zeros([2]), name="bias8")
    y_ = tf.matmul(relu7, weights8) + bias8


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

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y))

tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, trainable=False)
# It was trained using mini-batch stochastic gradient descent with "inv"
# learning rate starting from 0.001 (power: 0.75; gamma: 0.0001;weight_decay: 0.0005)
# and a momentum fixed to 0.9
# - inv: base_lr * (1 + gamma * iter) ^ (- power)
learning_rate = starter_learning_rate*(1+0.0001*tf.cast(global_step, dtype=tf.float32))**(-decay_rate)
opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=0.9, epsilon=5e-04).minimize(loss, var_list=params, global_step=global_step)

data_x = np.zeros([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
data_y = np.zeros([BATCH_SIZE, NUM_LABELS])

for i in range(0, BATCH_COVER):
    data_y[i, 1] = 1
for i in range(BATCH_COVER, BATCH_SIZE):
    data_y[i, 0] = 1

saver = tf.train.Saver(max_to_keep=10000)
merged = tf.summary.merge_all()
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True

with tf.Session() as sess:
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    tf.global_variables_initializer().run()

    if NUM_ITER > 0:
        saver.restore(sess, model_dir + '/' + str(NUM_ITER) + '.ckpt')
        tf.assign(global_step, NUM_ITER)

    summary = tf.Summary()

    count = 0
    random.shuffle(fileList)
    for i in range(NUM_ITER, MAX_ITER+1):
        # print('item: %d, batch: %d, num: %d' % (i,BATCH_SIZE,NUM_ITER))
        for j in range(BATCH_COVER):
            if count >= NUM_SAMPLE:
                count = 0
                random.shuffle(fileList)

            cover, stego = read_ppm(cover_dir, stego_dir, fileList[count])
            data_x[j, :, :, :] = cover.astype(np.float32)
            data_x[j + BATCH_COVER, :, :, :] = stego.astype(np.float32)
            count = count+1

        _, tempL, tempA, tempC, tempS, tempY = sess.run([opt, loss, accuracy, accuracyCover, accuracyStego, y_],
                                              feed_dict={x: data_x, y: data_y, is_train: True})

        if i % NUM_SHOWTRAIN == 0:
            summary.ParseFromString(sess.run(merged, feed_dict={x: data_x, y: data_y, is_train: True}))
            writer.add_summary(summary, i)
            print('WISERENet, DB: %(d)s, Cost: %(c)s, payload: %(p).2f batch results:' %
                            {'d': DB_SET, 'c': COST_FUNC, 'p': PAYLOAD})
            print('iter=%(i)d, loss=%(s).4f, cover_acc=%(c).4f, stego_acc=%(t).4f, accuracy=%(a).4f...\n' %
                            {'i': i, 's': tempL, 'a': tempA, 'c': tempC, 't': tempS})

        if i % NUM_SHOWTEST == 0:
            saver.save(sess, model_dir+'/'+str(i)+'.ckpt')

            # validation
            loss1 = np.array([])
            acc1 = np.array([])  # accuracy for validation set
            cover1 = np.array([])
            stego1 = np.array([])

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

                    cover, stego = read_ppm(validation_cover, validation_stego, validationList[val_count])
                    data_x[j, :, :, :] = cover.astype(np.float32)
                    data_x[j + BATCH_COVER, :, :, :] = stego.astype(np.float32)
                    val_count = val_count + 1

                tempL, tempA, tempC, tempS = sess.run([loss, accuracy, accuracyCover, accuracyStego],
                                                      feed_dict={x: data_x, y: data_y, is_train: False})
                loss1 = np.append(loss1, tempL)
                acc1 = np.append(acc1, tempA)
                cover1 = np.append(cover1, tempC)
                stego1 = np.append(stego1, tempS)

            loss_val = np.mean(loss1)
            acc_val = np.mean(acc1)
            acc_cover_val = np.mean(cover1)
            acc_stego_val = np.mean(stego1)
            summary.value.add(tag='val_loss', simple_value=loss_val)
            summary.value.add(tag='val_acc', simple_value=acc_val)
            summary.value.add(tag='val_acc_cover', simple_value=acc_cover_val)
            summary.value.add(tag='val_acc_stego', simple_value=acc_stego_val)
            writer.add_summary(summary, i)
            print('Validation: iter = %(i)d, loss = %(s).4f, acc_cover = %(c).4f, acc_stego = %(t).4f, accuracy = %(a).4f...' %
                        {'i': i, 's': loss_val, 'a': acc_val, 'c': acc_cover_val, 't': acc_stego_val})
            print('--------------\n')
            ErrCover = 1-acc_cover_val
            ErrStego = 1-acc_stego_val
            ErrRate = 1-acc_val
            with open(validation_result, 'a+') as f_out:
                f_out.write('%d,%.4f,%.4f,%.4f,%.4f,%.4f\n' % (i, loss_val, ErrCover, ErrStego, ErrRate,
                                                               abs(ErrCover-ErrStego)/ErrRate))

