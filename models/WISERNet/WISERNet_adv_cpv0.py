# # WISERNet: Wider Separate-Then-Reunion Network for Steganalysis of Color Images
# # the config file contains sample dir, model dir, eta.
# # 2020.12.30. use CPV.
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os
import random
from scipy import ndimage
import scipy.io as sio
import sys
import time
import getopt
import ConfigParser as configparser
# import configparser
sys.path.append('./tflib/')        # path for 'tflib' folder
from generator import *

BATCH_COVER = 1
BATCH_SIZE = 1
IMAGE_SIZE = 256  # original: 512
NUM_CHANNEL = 3
NUM_LABELS = 2
NUM_ITER = 0
# MAX_ITER = 300000
# NUM_SHOWTRAIN = 100 #show result eveary epoch
# NUM_SHOWTEST = 10000

N_MAGNIFICATION = 9

IS_STC = 0  # 1: use STC; otherwise use simulator.
STC_H = 10  # STC parameter

model_dir = ''
cover_dir = ''
stego_dir = ''
validation_cover = ''
validation_stego = ''
validation_proc = ''
sample_file = 'boss256_color_trn.txt'
config_file = '/data1/dataset/BOSS256_C/config/adv_WISERNet_HILL_BOSS256_PPM_p40_a01_val.cfg'

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
# visible_devices = config_raw.get('Environment', 'CUDA_VISIBLE_DEVICES')
# BATCH_COVER = config_raw.getint('Basic', 'BATCH_COVER')
# BATCH_SIZE = 2*BATCH_COVER
IMAGE_SIZE = config_raw.getint('Basic', 'IMAGE_SIZE')  # 256
# NUM_CHANNEL = 1
# NUM_LABELS = 2
NUM_ITER = config_raw.getint('Basic', 'NUM_ITER')
# MAX_ITER = config_raw.getint('Basic', 'MAX_ITER')
# NUM_SHOWTRAIN = config_raw.getint('Basic', 'NUM_SHOWTRAIN')
# NUM_SHOWTEST = config_raw.getint('Basic', 'NUM_SHOWTEST')

DIS_VER = config_raw.getint('Basic', 'DIS_VER')
DB_SET = config_raw.get('Basic', 'DB_SET')
COST_FUNC = config_raw.get('Basic', 'COST_FUNC')
PAYLOAD = config_raw.getfloat('Basic', 'PAYLOAD')

# starter_learning_rate = config_raw.getfloat('Basic', 'starter_learning_rate')  # 0.001
# decay_steps = config_raw.getfloat('Basic', 'decay_steps')  # 5000
# decay_rate = config_raw.getfloat('Basic', 'decay_rate')  # 0.75
if config_raw.has_option('Basic', 'IS_STC'):
    IS_STC = config_raw.getint('Basic', 'IS_STC')
if IS_STC == 1:
    if config_raw.has_option('Basic', 'STC_H'):
        STC_H = config_raw.getint('Basic', 'STC_H')
    import matlab.engine
    eng = matlab.engine.start_matlab()

model_dir = config_raw.get('Path', 'model_dir')
cover_dir = config_raw.get('Path', 'cover_dir')
cost_dir = config_raw.get('Path', 'cost_dir')
stego_dir = config_raw.get('Path', 'stego_dir')
sample_file = config_raw.get('Path', 'sample_file')
result_file = config_raw.get('Path', 'result_file')
check_stego = 0  # [2021.01.27]1: if the stego exists and can not deceive the CNN, reproduce it again.
if config_raw.has_option('Path', 'check_stego'):
    check_stego = config_raw.getint('Path', 'check_stego')

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(memory_gpu))
os.system('rm tmp')

if not os.path.exists(stego_dir):
    os.mkdir(stego_dir)

fileList = []
if len(sample_file) > 0:
    with open(sample_file, 'r') as f:
        lines = f.readlines()
        fileList = fileList + [a.strip() for a in lines]
else:
    for (dirpath, dirnames, filenames) in os.walk(stego_dir):
        fileList = filenames
NUM_SAMPLE = len(fileList)

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

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y))

grad = tf.gradients(loss, x)
s = tf.sign(grad)

data_x = np.zeros([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
data_y = np.zeros([BATCH_SIZE, NUM_LABELS])

for i in range(0, BATCH_SIZE):
    data_y[i, 0] = 1

config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True


def EmbeddingSTC_CPV(X, rhoCPV, payload_rate):  # embedding by STC.
    # changes = np.array([[1, 1, 1], [-1, -1, -1], [1, 1, 0], [1, 0, 1], [1, 0, 0],
    #                     [-1, -1, 0], [-1, 0, -1], [-1, 0, 0], [0, 1, 1], [0, 1, 0],
    #                     [0, -1, -1], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 1, -1],
    #                     [1, -1, 1], [1, -1, -1], [1, -1, 0], [1, 0, -1], [-1, 1, 1],
    #                     [-1, 1, -1], [-1, 1, 0], [-1, -1, 1], [-1, 0, 1], [0, 1, -1],
    #                     [0, -1, 1], [0, 0, 0]], dtype=np.int)
    # rho_0 = rhoCPV.reshape(-1, 3)
    # rho_stc = np.zeros([rho_0.shape[0], 1, 1, 1], dtype=np.float32)
    # for idx_r in range(27):
    #     rho_stc[:, changes[idx_r, 0], changes[idx_r, 1], changes[idx_r, 2]] = rho_0[:, idx_r]
    x_0 = matlab.int32(X.tolist())
    rho_stc = matlab.single(rhoCPV.tolist())  # matlab.single(rho_stc.tolist())
    payload_rate = float(payload_rate)
    # D, Y, NUM_MSG_BITs, L, USE_STC = stc_embed_wpv(X, COSTs, MSG, H, verify_extra)
    y_0 = eng.stc_embed_cpv(x_0, rho_stc, payload_rate, nargout=2)
    Y = np.array(y_0[0])
    stc_code = y_0[1]
    modification = Y.astype(np.float32)-X.astype(np.float32)
    return modification, stc_code


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    saver.restore(sess, model_dir + '/' + str(NUM_ITER) + '.ckpt')

    Loss = np.array([])
    Accuracy = np.array([])

    ALPHA = 2.0
    WetCost = 1e13
    MESSAGE_LENGTH = PAYLOAD*IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNEL

    if not os.path.exists(result_file):
        with open(result_file, 'w+') as f_s:
            f_s.write('WISERNet,%s,%s,%.2f\nmodel:,%s\n' % (DB_SET, COST_FUNC, PAYLOAD, model_dir))
            f_s.write('sample:,%s\n' % sample_file)
            f_s.write('stego:,%s\n' % stego_dir)
            f_s.write('item,image,success,sigma,loss0,loss,diff0,diff,stc,time\n')
    count = 0
    while count < NUM_SAMPLE:
        # [2021.01.27]if the stego exists and cannot deceive the CNN, reproduce it again.
        if (check_stego == 1) & (os.path.exists(stego_dir+'/'+fileList[count])):
            coefS = read_single_ppm(stego_dir, fileList[count])
            data_x[0, :, :, :] = coefS.astype(np.float32)
            tempA, temp0, tempS = sess.run([accuracy, loss, s], feed_dict={x: data_x, y: data_y, is_train: False})
            if temp0 > 0.7:
                count += 1
                continue

        coefC = read_single_ppm(cover_dir, fileList[count])
        rhoCPV = read_cpv_cost(cost_dir, fileList[count], IMAGE_SIZE)
        stc_code = IS_STC

        print('count: %d, img: %s, net: WISERNet, cost: %s, payload: %.2f' % (count, fileList[count], COST_FUNC, PAYLOAD))

        time_0 = time.time()

        sigma_val = 0
        stego_0 = coefC.copy()
        loss_0 = 0
        dif_0 = 0
        dif_cs = 0
        coefS = coefC.astype(np.int32)
        temp0 = 0
        tempA = 0
        np.random.seed(5678)
        randMatrix = np.random.rand(coefC.shape[0], coefC.shape[1])
        for SIGMA in np.arange(0, 1.001, 0.1):
            coefS = coefC.astype(np.int32)
            rhoCPV_n = rhoCPV.copy()
            sigma_val = SIGMA
            stc_code = IS_STC
            for idx_cpv in range(26):
                rhoCPV_c = rhoCPV_n[:, :, idx_cpv]
                rhoCPV_c[randMatrix < SIGMA] = WetCost  # (SIGMA) DCT is set as wetCost; (1-SIGMA) DCT for the first embedding
                rhoCPV_n[:, :, idx_cpv] = rhoCPV_c

            if (IS_STC == 1) & (stc_code == 1):
                m_block, stc_code = EmbeddingSTC_CPV(coefC, rhoCPV_n, PAYLOAD*(1-SIGMA))
            else:
                m_block = embed_cpv(coefC, rhoCPV_n, round(MESSAGE_LENGTH * (1 - SIGMA)))

            coefS += m_block.astype(np.int32)

            data_x[0, :, :, :] = coefS.astype(np.float32)
            tempA, temp0, tempS = sess.run([accuracy, loss, s], feed_dict={x: data_x, y: data_y, is_train: False})
            tempS = tempS.reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL)

            if SIGMA == 0:
                stego_0 = coefS.copy()
                loss_0 = temp0
                dif_0 = np.sum(coefC != coefS)
                dif_cs = dif_0
            else:
                rhoCPV_n = adjust_cpv_cost(rhoCPV, tempS, 1/ALPHA, 0)  # 1/ALPHA: the function is implemented for ITE.
                for idx_cpv in range(26):
                    rhoCPV_c = rhoCPV_n[:, :, idx_cpv]
                    rhoCPV_c[randMatrix >= SIGMA] = WetCost  # (SIGMA) DCT is set as wetCost; (1-SIGMA) DCT for the first embedding
                    rhoCPV_n[:, :, idx_cpv] = rhoCPV_c
                if (IS_STC == 1) & (stc_code == 1):
                    m_block, stc_code = EmbeddingSTC_CPV(coefC, rhoCPV_n, PAYLOAD * SIGMA)
                else:
                    if IS_STC == 1:
                        m_block, stc_code = EmbeddingSTC_CPV(coefC, rhoCPV_n, PAYLOAD * SIGMA*1.05)
                    else:
                        m_block = embed_cpv(coefC, rhoCPV_n, round(MESSAGE_LENGTH * SIGMA))
                coefS += m_block.astype(np.int32)

                data_x[0, :, :, :] = coefS.astype(np.float32)
                tempA, temp0, tempS = sess.run([accuracy, loss, s], feed_dict={x: data_x, y: data_y, is_train: False})
                dif_cs = np.sum(coefC != coefS)

            print('STC: %d, sigma: %.2f, acc: %.4f, loss: %.4f, dif: %d' % (stc_code, SIGMA, tempA, temp0, dif_cs))

            if temp0 > 0.7:
                break

        time_s = time.time() - time_0
        save_ppm(coefS.astype(np.uint8), stego_dir, fileList[count])
        if temp0 > 0.7:  # Adversarial examples can fool detector
            # sio.savemat(stego_dir+'/'+fileList[count], {'coefC': coefC, 'coefS': coefS, 'LossPerImage': LossPerImage})
            with open(result_file, 'a+') as f_s:
                f_s.write('%d,%s,1,%.2f,%.4f,%.4f,%d,%d,%d,%.4f\n' %
                          (count, fileList[count], sigma_val, loss_0, temp0, dif_0, dif_cs, stc_code, time_s))
        else:
            # m_block = EmbeddingSimulator(rhoP1, rhoM1, round(PAYLOAD*IMAGE_SIZE*IMAGE_SIZE))
            # coefS = coefC+m_block
            coefS = stego_0
            dif_cs = dif_0
            # sio.savemat(stego_dir+'/'+fileList[count], {'coefC': coefC, 'coefS': coefS, 'LossPerImage': LossPerImage})
            with open(result_file, 'a+') as f_s:
                f_s.write('%d,%s,0,%.2f,%.4f,%.4f,%d,%d,%d,%.4f\n' %
                          (count, fileList[count], sigma_val, loss_0, temp0, dif_0, dif_cs, stc_code, time_s))

        print('acc: %.4f, loss: %.4f, dif: %d, time: %.4fs' % (tempA, temp0, dif_cs, time_s))
        Accuracy = np.insert(Accuracy, 0, tempA)
        Loss = np.insert(Loss, 0, temp0)
        # sio.savemat(pathO+'/'+fileList[count],{'spatC':spatC,'spatS':spatS})
        count = count + 1

    print("final result")
    print('Acc: %.4f, loss: %.4f\n' % (np.mean(Accuracy), np.mean(Loss)))

