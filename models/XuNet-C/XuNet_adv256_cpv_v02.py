# #XuNet (with 6 groups).
# #Date: 2019.09.09.
# #Use the config file to import parameters.
# #Update:2019.12.20. Add STC mode.

import sys
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os
import random
import math
import h5py
from scipy import ndimage
from scipy import signal
import scipy.io as sio
import getopt
import ConfigParser as configparser
# import configparser
import time
sys.path.append('./tflib/')        # path for 'tflib' folder
from generator import *

BATCH_SIZE = 1
IMAGE_SIZE = 256
NUM_CHANNEL = 3
NUM_LABELS = 2
# NUM_ITER = 230000
NUM_ITER = 200000
NUM_SHOWTRAIN = 100 #show result eveary epoch
NUM_SHOWTEST = 100
PI = math.pi
COST_FUNC = 'CPV'
PAYLOAD = 0.4
DIS_VER = 0
DB_SET = 'BOSS256_C'

IS_STC = 0  # 1: use STC; otherwise use simulator.
STC_H = 10  # STC parameter

sample_file = ''
config_file = '/pubdata/qxh/BOSS256_C/config/adv_XuNet_CPV_i104k_BOSS256_PPM_p40_a01_val.cfg'

opts, args = getopt.getopt(sys.argv[1:], 'hc:',	['help', 'config='])

for opt, val in opts:
	if opt == '-c':
		config_file = val
	if opt in ('-h', '--help'):
		print('Create adversarial samples by using XuNet (with 6 CNN groups).')
		print('  -c: config file.')
		sys.exit()
# Read configs.
config_raw = configparser.RawConfigParser()
config_raw.read(config_file)
# visible_device = config_raw.get('Environment', 'CUDA_VISIBLE_DEVICES')
NUM_ITER = config_raw.getint('Basic', 'NUM_ITER')  #200000
PAYLOAD = config_raw.getfloat('Basic', 'PAYLOAD')  #0.4
# BETA = config_raw.getfloat('Basic', 'BETA')  #0.1
# GAMMA = config_raw.getfloat('Basic', 'GAMMA')  #0.01
# STEP = config_raw.getfloat('Basic', 'STEP')  #0.05
# MARGINE = config_raw.getfloat('Basic', 'MARGINE')  #0.01
CMD_FACTOR = 0  # 0.11
if config_raw.has_option('Basic', 'CMD_FACTOR'):
	CMD_FACTOR = config_raw.getfloat('Basic', 'CMD_FACTOR')
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

cost_version = 0
if config_raw.has_option('Basic', 'COST_VERSION'):
	cost_version = config_raw.getfloat('Basic', 'COST_VERSION')

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

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = visible_device  # '4'
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(memory_gpu))
os.system('rm tmp')

if not os.path.exists(stego_dir):
	os.mkdir(stego_dir)

fileList = []
# sigma_file = ''
if len(sample_file) > 0:
	with open(sample_file, 'r') as f:
		lines = f.readlines()
		fileList = fileList + [a.strip() for a in lines]

x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
y = tf.placeholder(tf.float32, shape=[BATCH_SIZE,NUM_LABELS])
is_train = tf.placeholder(tf.bool, name='is_train')

hpf = np.zeros([5, 5, NUM_CHANNEL, 1], dtype=np.float32)
for i in range(NUM_CHANNEL):
	hpf[:, :, i, 0] = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
								[2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32)/12

kernel0 = tf.Variable(hpf, name="kernel0")
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

	relu3 = tf.nn.relu(bn3, name="bn3")
	pool3 = tf.nn.avg_pool(relu3, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool3")

with tf.variable_scope("Group4") as scope:
	kernel4_1 = tf.Variable(tf.random_normal([1, 1, 96, 192], mean=0.0, stddev=0.01), name="kernel4_1")
	conv4_1 = tf.nn.conv2d(pool3, kernel4_1, [1, 1, 1, 1], padding="SAME", name="conv4_1")

	bn4_1 = slim.layers.batch_norm(conv4_1, is_training=is_train, updates_collections=None, decay=0.05)
	relu4_1 = tf.nn.relu(bn4_1, name="relu4_1")
	pool4 = tf.nn.avg_pool(relu4_1, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool4_1")

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
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_))

grad = tf.gradients(loss, x)
s = tf.sign(grad)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 5000, 0.9, staircase=True)
opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss, var_list=params, global_step=global_step)

data_x = np.zeros([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
data_y = np.zeros([BATCH_SIZE, NUM_LABELS])

# for i in range(0, BATCH_SIZE):
# 	data_y[i, 0] = 1
data_y[:, 1] = 1  # cover label

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
	# D, Y, NUM_MSG_BITs, L, USE_STC = stc_embed_wpv(X, COSTs, MSG, H, verify_extra)
	y_0 = eng.stc_embed_wpv(x_0, rho_stc, payload_rate, nargout=2)
	Y = np.array(y_0[0])
	stc_code = y_0[1]
	modification = Y.astype(np.float32)-X.astype(np.float32)
	return modification, stc_code


with tf.Session(config=config) as sess:
	tf.global_variables_initializer().run()

	saver = tf.train.Saver()
	saver.restore(sess, model_dir+'/'+str(NUM_ITER)+'.ckpt')

	Loss = np.array([])
	Accuracy = np.array([])

	# BLOCK_STRIDE = 2   # 2 4 8
	# ALPHA = 2
	# SUBIMAGE_SIZE = IMAGE_SIZE/BLOCK_STRIDE
	WetCost = 1e13
	bs_index = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int)

	if not os.path.exists(result_file):
		with open(result_file, 'w+') as f_s:
			f_s.write('model:,%s\n' % model_dir)
			f_s.write('sample:,%s\n' % sample_file)
			f_s.write('stego:,%s\n' % stego_dir)
			f_s.write('item,image,success,alpha,iter,loss0,loss,diff0,diff,stc,time\n')
	len_block_msg = round(PAYLOAD * IMAGE_SIZE * IMAGE_SIZE*NUM_CHANNEL / 4)  # split to 4 blocks.
	if IS_STC == 1:
		len_block_msg = round(1.05*len_block_msg)  # add 0.05 payload to trade-off that by SIM.
	count = 0
	while count < len(fileList):
		print('count: %d, image: %s, payload: %.2f, cmd: %.2f' % (count, fileList[count], PAYLOAD, CMD_FACTOR))
		# dataC = h5py.File(pathI+fileList[count].replace('.pgm', '.mat'))
		# dataC = sio.loadmat(cover_dir + '/' + fileList[count])
		# coefC = dataC['coef']
		# rhoP1 = dataC['rhoP1']
		# rhoM1 = dataC['rhoM1']
		coefC = read_single_ppm(cover_dir, fileList[count])
		rhoCPV = read_cpv_cost(cost_dir, fileList[count], IMAGE_SIZE)
		stc_code = IS_STC

		time_0 = time.time()
		# LossPerImage = np.array([])

		coefS = coefC.copy()
		ALPHA = 1  # ALPHA = 1+ETA
		b_index = 0  # random.randint(0, 3)
		rhoCPV_n = rhoCPV.copy()
		for i in range(4):
			# b_index = (b_index + 1) % 4
			r_0 = bs_index[i, 0]  # bs_index[b_index, 0]
			c_0 = bs_index[i, 1]  # bs_index[b_index, 1]
			rho_b = rhoCPV_n[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2, :]
			x_0 = coefS[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2, :]
			if (IS_STC == 1) & (stc_code == 1):
				b_block, stc_code = EmbeddingSTC_CPV(x_0, rho_b, PAYLOAD)
			else:
				b_block = embed_cpv(x_0, rho_b, len_block_msg)

			y_0 = x_0 + b_block
			# coefS[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2, :] = coefS[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2, :] + b_block
			coefS[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2, :] = y_0

		# discriminate.
		data_x[0, :, :, :] = coefS.astype(np.float32)
		tempA, temp0 = sess.run([accuracy, loss], feed_dict={x: data_x, y: data_y, is_train: False})
		dif_emb = coefS.astype(np.int32) - coefC.astype(np.int32)
		dif_0 = np.abs(dif_emb).sum()
		print('STC: %d, alpha: %.4f, acc: %.4f, loss: %.4f, dif: %d' % (stc_code, ALPHA, tempA, temp0, dif_0))
		# LossPerImage = np.append(LossPerImage, temp0)
		loss_0 = temp0
		dif_cs = dif_0
		iter_no = 0
		if tempA < 1:  # discriminate correct.
			stego_ite = coefS.copy()
			ZETA = DELTA_ZETA0
			DELTA_ZETA = DELTA_ZETA0
			b_index0 = random.randint(0, 3)
			# grad_s = tempS.reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL)
			# for ZETA in np.arange(DELTA_ZETA, MAX_ZETA+DELTA_ZETA, DELTA_ZETA):
			while ZETA < MAX_ZETA+0.01:
			# while iter_no < MAX_ITER:
				if tempA >= 1:
					break
				ALPHA = 1 + ZETA
				if ZETA > INTER_ZETA:
					DELTA_ZETA = DELTA_ZETA1
				ZETA += DELTA_ZETA
				# grad_s = tempS.reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL)
				for i in range(4):
					b_index = (b_index0 + i) % 4
					r_0 = bs_index[b_index, 0]
					c_0 = bs_index[b_index, 1]
					# initialize the block values.
					b_cover = coefC[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2, :]
					if CMD_FACTOR > 0:  # embed message with CMD.
						rhoCPV_n = adjust_cmd_cpv(rhoCPV, coefC, coefS, CMD_FACTOR)
						b_rho = rhoCPV_n[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2, :]
						if (IS_STC == 1) & (stc_code == 1):
							b_block, stc_code = EmbeddingSTC_CPV(b_cover, b_rho, PAYLOAD)
						else:
							b_block = embed_cpv(b_cover, b_rho, len_block_msg)
						b_stego = b_cover + b_block
						stego_ite[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2, :] = b_stego
						data_x[0, :, :, :] = stego_ite.astype(np.float32)
						tempA, temp0 = sess.run([accuracy, loss], feed_dict={x: data_x, y: data_y, is_train: False})
						if tempA >= 1:
							dif_emb = stego_ite.astype(np.int32) - coefC.astype(np.int32)
							dif_ite = np.abs(dif_emb).sum()  # np.sum(coefC != stego_ite)
							coefS = stego_ite
							dif_cs = dif_ite
							break
					# resume the pending part to cover.
					stego_ite[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2, :] = b_cover
					data_x[0, :, :, :] = stego_ite.astype(np.float32)
					tempS = sess.run(s, feed_dict={x: data_x, y: data_y, is_train: False})
					grad_s = tempS.reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL)
					# ---------
					b_sign = grad_s[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2, :]
					b_rho = rhoCPV_n[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2, :]
					b_rho = adjust_cpv_cost(b_rho, b_sign, ALPHA, cost_version)
					if (IS_STC == 1) & (stc_code == 1):
						b_block, stc_code = EmbeddingSTC_CPV(b_cover, b_rho, PAYLOAD)
					else:
						b_block = embed_cpv(b_cover, b_rho, len_block_msg)
					b_stego = b_cover + b_block
					stego_ite[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2, :] = b_stego
					data_x[0, :, :, :] = stego_ite.astype(np.float32)
					tempA, temp0 = sess.run([accuracy, loss], feed_dict={x: data_x, y: data_y, is_train: False})
					dif_emb = coefC != stego_ite
					dif_ite = dif_emb.sum()  # np.sum(coefC != stego_ite)
					iter_no += 1
					print('iter: %d, STC: %d, alpha: %.4f, acc: %.4f, loss: %.4f, dif: %d' %
															(iter_no, stc_code, ALPHA, tempA, temp0, dif_ite))
					# LossPerImage = np.append(LossPerImage, temp0)
					if tempA >= 1:
						coefS = stego_ite
						dif_cs = dif_ite
						break

		time_s = time.time()-time_0
		save_ppm(coefS, stego_dir, fileList[count])
		if tempA >= 1:  # Adversarial examples can fool detector
			# f_s.write('item,image,success,alpha,iter,loss0,loss,diff0,diff,stc,time\n')
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
