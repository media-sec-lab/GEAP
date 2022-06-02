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

BATCH_COVER = 1
BATCH_SIZE = 1
IMAGE_SIZE = 256
NUM_CHANNEL = 3
NUM_LABELS = 2
NUM_ITER = 0

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
config_file = '/data1/dataset/BOSS256_C/config/ite_WISERNet_HILL_BOSS256_PPM_p40_a01_val.cfg'

opts, args = getopt.getopt(sys.argv[1:], 'hc:', ['help', 'config='])

for opt, val in opts:
    if opt == '-c':
        config_file = val
    if opt in ('-h', '--help'):
        print('element-GEAP attacks WISERNet.')
        print('  -c: the config file.')
        sys.exit()

# Read configs.
config_raw = configparser.RawConfigParser()
config_raw.read(config_file)
IMAGE_SIZE = config_raw.getint('Basic', 'IMAGE_SIZE')  # 256
NUM_ITER = config_raw.getint('Basic', 'NUM_ITER')

DIS_VER = config_raw.getint('Basic', 'DIS_VER')
DB_SET = config_raw.get('Basic', 'DB_SET')
COST_FUNC = config_raw.get('Basic', 'COST_FUNC')
PAYLOAD = config_raw.getfloat('Basic', 'PAYLOAD')
MAX_ZETA = 2.0
DELTA_ZETA0 = 0.01
INTER_ZETA = 1.0
DELTA_ZETA1 = 0.1
if config_raw.has_option('Basic', 'MAX_ZETA'):
    MAX_ZETA = config_raw.getfloat('Basic', 'MAX_ZETA')
if config_raw.has_option('Basic', 'DELTA_ZETA0'):
    DELTA_ZETA0 = config_raw.getfloat('Basic', 'DELTA_ZETA0')
if config_raw.has_option('Basic', 'DELTA_ZETA1'):
    DELTA_ZETA1 = config_raw.getfloat('Basic', 'DELTA_ZETA1')
if config_raw.has_option('Basic', 'INTER_ZETA'):
    INTER_ZETA = config_raw.getfloat('Basic', 'INTER_ZETA')

if config_raw.has_option('Basic', 'IS_STC'):
    IS_STC = config_raw.getint('Basic', 'IS_STC')
if IS_STC == 1:
    if config_raw.has_option('Basic', 'STC_H'):
        STC_H = config_raw.getint('Basic', 'STC_H')
    import matlab.engine
    eng = matlab.engine.start_matlab()
cost_version = 0  # 1: entire vector.
if config_raw.has_option('Basic', 'COST_VERSION'):
    cost_version = config_raw.getfloat('Basic', 'COST_VERSION')

L1 = 2
W1 = 2
if config_raw.has_option('Basic', 'L1'):
    L1 = config_raw.getint('Basic', 'L1')
if config_raw.has_option('Basic', 'W1'):
    W1 = config_raw.getint('Basic', 'W1')

model_dir = config_raw.get('Path', 'model_dir')
cover_dir = config_raw.get('Path', 'cover_dir')
cost_dir = config_raw.get('Path', 'cost_dir')
stego_dir = config_raw.get('Path', 'stego_dir')
sample_file = config_raw.get('Path', 'sample_file')
result_file = config_raw.get('Path', 'result_file')

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

DECAY_RATE = 0.95
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

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y))

grad = tf.gradients(loss, x)
s = tf.sign(grad)

data_x = np.zeros([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
data_y = np.zeros([BATCH_SIZE, NUM_LABELS])

for i in range(0, BATCH_SIZE):
    data_y[i, 1] = 1

config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True


def EmbeddingSTC_CPV(X, rhoCPV, payload_rate):  # embedding by STC.
    changes = np.array([[1, 1, 1], [-1, -1, -1], [1, 1, 0], [1, 0, 1], [1, 0, 0],
                        [-1, -1, 0], [-1, 0, -1], [-1, 0, 0], [0, 1, 1], [0, 1, 0],
                        [0, -1, -1], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 1, -1],
                        [1, -1, 1], [1, -1, -1], [1, -1, 0], [1, 0, -1], [-1, 1, 1],
                        [-1, 1, -1], [-1, 1, 0], [-1, -1, 1], [-1, 0, 1], [0, 1, -1],
                        [0, -1, 1], [0, 0, 0]], dtype=np.int)
    rho_0 = rhoCPV.reshape(-1, 3)
    rho_stc = np.zeros([rho_0.shape[0], 1, 1, 1], dtype=np.float32)
    for idx_r in range(27):
        rho_stc[:, changes[idx_r, 0], changes[idx_r, 1], changes[idx_r, 2]] = rho_0[:, idx_r]
    x_0 = matlab.int32(X.tolist())
    rho_stc = matlab.single(rho_stc.tolist())
    payload_rate = float(payload_rate)
    y_0 = eng.stc_embed_wpv(x_0, rho_stc, payload_rate, nargout=2)
    Y = np.array(y_0[0])
    stc_code = y_0[1]
    modification = Y.astype(np.float32)-X.astype(np.float32)
    return modification, stc_code


def EmbeddingSimulator(rhoP, rhoM, m):
    n = rhoP.size
    Lambda = calc_lambda(rhoP, rhoM, m, n)

    zp = np.exp(-Lambda * rhoP)
    zm = np.exp(-Lambda * rhoM)
    z0 = 1 + zp + zm
    pChangeP1 = zp / z0
    pChangeM1 = zm / z0

    randChange = np.random.rand(rhoP.shape[0], rhoP.shape[1])
    modification = np.zeros([rhoP.shape[0], rhoP.shape[1]])
    modification[randChange < pChangeP1] = 1
    modification[randChange >= 1 - pChangeM1] = -1
    return modification


# -, +, the bits that should be embedded, number of image
def calc_lambda(rhoP, rhoM, message_length, n):
    l3 = 1e+3
    m3 = message_length + 1
    iterations = 0
    while m3 > message_length:
        l3 = l3 * 2
        zp = np.exp(-l3 * rhoP)
        zm = np.exp(-l3 * rhoM)
        z0 = 1 + zp + zm
        pP1 = zp / z0
        pM1 = zm / z0
        m3 = ternary_entropyf(pP1, pM1)
        iterations = iterations + 1
        if iterations > 10:
            Lambda = l3
            return Lambda

    l1 = 0
    m1 = n
    Lambda = 0

    alpha = float(message_length) / n
    while (float(m1 - m3) / n > alpha / 1000.0) and (iterations < 30):
        Lambda = l1 + (l3 - l1) / 2.0
        zp = np.exp(-Lambda * rhoP)
        zm = np.exp(-Lambda * rhoM)
        z0 = 1 + zp + zm
        pP1 = zp / z0
        pM1 = zm / z0
        m2 = ternary_entropyf(pP1, pM1)
        if m2 < message_length:
            l3 = Lambda
            m3 = m2
        else:
            l1 = Lambda
            m1 = m2
        iterations = iterations + 1
    return Lambda


def ternary_entropyf(pP1, pM1):
    p0 = 1 - pP1 - pM1
    p0[p0 == 0] = 1e-10
    pP1[pP1 == 0] = 1e-10
    pM1[pM1 == 0] = 1e-10

    P = np.array([p0, pP1, pM1]).flatten()
    H = - (P * np.log2(P))
    H[(P < 2.2204e-16) | (P > 1 - 2.2204e-16)] = 0

    Ht = sum(H)
    return Ht


def EmbeddingSTC(X, rhoP, rhoM, payload_rate, stc_h):  # embedding by STC.
    x_0 = matlab.int32(X.tolist())
    rho_p = matlab.single(rhoP.tolist())
    rho_m = matlab.single(rhoM.tolist())
    payload_rate = float(payload_rate)
    y_0 = eng.stc_embedding(x_0, rho_p, rho_m, payload_rate, stc_h, nargout=2)
    Y = np.array(y_0[0])
    stc_c = y_0[1]
    modification = Y.astype(np.float32) - X.astype(np.float32)
    return modification, stc_c


def sim_embedd_color(rho_p, rho_m, len_msg_c):
    m_e = np.zeros(rho_p.shape)
    for idx_c in range(NUM_CHANNEL):
        m_i = EmbeddingSimulator(rho_p[:, :, idx_c], rho_m[:, :, idx_c], len_msg_c)
        m_e[:, :, idx_c] = m_i
    return m_e


def stc_embedd_color(X, rho_p, rho_m, payload_rate, stc_h):
    m_e = np.zeros(rho_p.shape)
    stc_c = 1
    for idx_c in range(NUM_CHANNEL):
        m_i, stc_i = EmbeddingSTC(X[:, :, idx_c], rho_p[:, :, idx_c], rho_m[:, :, idx_c], payload_rate, stc_h)
        m_e[:, :, idx_c] = m_i
        if stc_i == 0:
            stc_c = 0
    return m_e, stc_c


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    saver.restore(sess, model_dir + '/' + str(NUM_ITER) + '.ckpt')

    Loss = np.array([])
    Accuracy = np.array([])

    WetCost = 1e13
    count_sub = L1*W1
    bs_index = np.zeros((count_sub, 2), dtype=np.int)
    b_idx = 0
    w_step = 1
    w_0 = 0
    w_1 = W1
    for u in range(L1):  # zig-zeg flow
        if w_step == 1:
            w_0 = 0
            w_1 = W1
        else:
            w_0 = W1-1
            w_1 = -1
        for v in range(w_0, w_1, w_step):
            bs_index[b_idx, 0] = u
            bs_index[b_idx, 1] = v
            b_idx += 1
        if w_step == 1:
            w_step = -1
        else:
            w_step = 1

    len_msg_block_channel = round(PAYLOAD * IMAGE_SIZE * IMAGE_SIZE / count_sub)

    with open(result_file, 'w+') as f_s:
        f_s.write('model:,%s\n' % model_dir)
        f_s.write('sample:,%s\n' % sample_file)
        f_s.write('stego:,%s\n' % stego_dir)
        f_s.write('item,image,success,alpha,iter,loss0,loss,diff0,diff,stc,time\n')
    count = 0
    while count < len(fileList):
        coefC = read_single_ppm(cover_dir, fileList[count])
        rhoP1, rhoM1 = read_cost(cost_dir, fileList[count])
        stc_code = IS_STC

        print('count: %d, WISERNet, %s, payload: %.2f, %s' % (count, COST_FUNC, PAYLOAD, fileList[count]))

        time_0 = time.time()

        b_index = 0
        coefS = coefC.copy()
        rhoP1N = rhoP1.copy()
        rhoM1N = rhoM1.copy()
        ALPHA = 1  # ALPHA = 1+ETA

        for i in range(count_sub):
            r_0 = bs_index[i, 0]
            c_0 = bs_index[i, 1]
            x_0 = coefC[r_0:IMAGE_SIZE:L1, c_0:IMAGE_SIZE:W1, :]
            rho_pb = rhoP1N[r_0:IMAGE_SIZE:L1, c_0:IMAGE_SIZE:W1, :]
            rho_mb = rhoM1N[r_0:IMAGE_SIZE:L1, c_0:IMAGE_SIZE:W1, :]
            if IS_STC == 1:
                m_block, stc_code = stc_embedd_color(x_0, rho_pb, rho_mb, PAYLOAD, STC_H)
            else:
                m_block = sim_embedd_color(rho_pb, rho_mb, len_msg_block_channel)
            coefS[r_0:IMAGE_SIZE:L1, c_0:IMAGE_SIZE:W1, :] = x_0 + m_block
            if CMD_FACTOR > 0:  # embed message with CMD.
                rhoP1N, rhoM1N = adjust_cmd_color(rhoP1, rhoM1, coefC, coefS, CMD_FACTOR)

        # discriminate.
        data_x[0, :, :, :] = coefS.astype(np.float32)
        tempA, temp0, tempS = sess.run([accuracy, loss, s], feed_dict={x: data_x, y: data_y, is_train: False})
        dif_emb = coefS.astype(np.int32) - coefC.astype(np.int32)
        dif_0 = np.abs(dif_emb).sum()
        print('STC: %d, alpha: %.4f, acc: %.4f, loss: %.4f, dif: %d' % (stc_code, ALPHA, tempA, temp0, dif_0))
        loss_0 = temp0
        dif_cs = dif_0
        iter_no = 0
        if tempA < 1:  # discriminate correct.
            stego_ite = coefS.copy()
            b_index0 = random.randint(0, count_sub)
            ZETA = DELTA_ZETA0
            DELTA_ZETA = DELTA_ZETA0
            while ZETA < MAX_ZETA+0.00001:
                if tempA >= 1:
                    break
                ALPHA = 1 + ZETA
                if ZETA > INTER_ZETA:
                    DELTA_ZETA = DELTA_ZETA1
                ZETA += DELTA_ZETA
                for i in range(count_sub):
                    b_index = (b_index0 + i) % count_sub
                    r_0 = bs_index[b_index, 0]
                    c_0 = bs_index[b_index, 1]
                    b_cover = coefC[r_0:IMAGE_SIZE:L1, c_0:IMAGE_SIZE:W1, :]

                    rho_pb = rhoP1N[r_0:IMAGE_SIZE:L1, c_0:IMAGE_SIZE:W1, :]
                    rho_mb = rhoM1N[r_0:IMAGE_SIZE:L1, c_0:IMAGE_SIZE:W1, :]
                    grad_s = tempS.reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL)
                    b_sign = grad_s[r_0:IMAGE_SIZE:L1, c_0:IMAGE_SIZE:W1, :]
                    rho_pb, rho_mb = adjust_adv_cost(rho_pb, rho_mb, b_sign, ALPHA, cost_version)

                    if IS_STC == 1:
                        m_block, stc_code = stc_embedd_color(b_cover, rho_pb, rho_mb, PAYLOAD, STC_H)
                    else:
                        m_block = sim_embedd_color(rho_pb, rho_mb, len_msg_block_channel)
                    stego_ite[r_0:IMAGE_SIZE:L1, c_0:IMAGE_SIZE:W1, :] = b_cover + m_block
                    data_x[0, :, :, :] = stego_ite.astype(np.float32)
                    tempA, temp0, tempS = sess.run([accuracy, loss, s],
                                                   feed_dict={x: data_x, y: data_y, is_train: False})
                    dif_emb = coefC != stego_ite
                    dif_ite = dif_emb.sum()
                    iter_no += 1
                    print('iter: %d, STC: %d, alpha: %.4f, acc: %.4f, loss: %.4f, dif: %d' %
                          (iter_no, stc_code, ALPHA, tempA, temp0, dif_ite))
                    if tempA >= 1:
                        coefS = stego_ite
                        dif_cs = dif_ite
                        break

        time_s = time.time() - time_0
        save_ppm(coefS, stego_dir, fileList[count])
        if tempA >= 1:  # Adversarial examples can fool the detector
            with open(result_file, 'a+') as f_s:
                f_s.write('%d,%s,1,%.4f,%d,%.4f,%.4f,%d,%d,%d,%.4f\n' %
                          (count, fileList[count], ALPHA, iter_no, loss_0, temp0, dif_0, dif_cs, stc_code, time_s))
        else:
            with open(result_file, 'a+') as f_s:
                f_s.write('%d,%s,0,%.4f,%d,%.4f,%.4f,%d,%d,%d,%.4f\n' %
                          (count, fileList[count], ALPHA, iter_no, loss_0, temp0, dif_0, dif_cs, stc_code, time_s))

        print('STC: %d, acc: %.4f, loss: %.4f, dif: %d, time: %.4fs' % (stc_code, tempA, temp0, dif_cs, time_s))
        Accuracy = np.insert(Accuracy, 0, tempA)
        Loss = np.insert(Loss, 0, temp0)
        count = count + 1

    print("final result")
    print('Acc: %.4f, loss: %.4f\n' % (np.mean(Accuracy), np.mean(Loss)))

