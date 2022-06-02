# #XuNet (with 6 groups).
# #Date: 2019.09.09.
# #Use the config file to import parameters.
# #Update: 2019.12.20. Add STC mode.

import sys
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os
import random
import math
import h5py
from scipy import ndimage
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
config_file = './config/adv_XuNet_HILLS0_i105k_BOSS_PPM256_p40_a01_val.cfg'

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
NUM_ITER = config_raw.getint('Basic', 'NUM_ITER')  # 200000
COST_FUNC = config_raw.get('Basic', 'COST_FUNC')
PAYLOAD = config_raw.getfloat('Basic', 'PAYLOAD')  # 0.4

if config_raw.has_option('Basic', 'IS_STC'):
	IS_STC = config_raw.getint('Basic', 'IS_STC')
if IS_STC == 1:
	if config_raw.has_option('Basic', 'STC_H'):
		STC_H = config_raw.getint('Basic', 'STC_H')
	import matlab.engine
	eng = matlab.engine.start_matlab()

# BETA = config_raw.getfloat('Basic', 'BETA')  #0.1
# GAMMA = config_raw.getfloat('Basic', 'GAMMA')  #0.01
# STEP = config_raw.getfloat('Basic', 'STEP')  #0.05
# MARGINE = config_raw.getfloat('Basic', 'MARGINE')  #0.01
model_dir = config_raw.get('Path', 'model_dir')
cover_dir = config_raw.get('Path', 'cover_dir')
cost_dir = config_raw.get('Path', 'cost_dir')
stego_dir = config_raw.get('Path', 'stego_dir')
sample_file = config_raw.get('Path', 'sample_file')
result_file = config_raw.get('Path', 'result_file')
check_stego = 0
if config_raw.has_option('Path', 'check_stego'):
	check_stego = config_raw.getint('Path', 'check_stego')

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

# c = np.load(model_dir+'/bn'+str(NUM_ITER)+'.npz')

x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_LABELS])
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

for i in range(0, BATCH_SIZE):
	data_y[i, 0] = 1

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

	ALPHA = 2.0
	WetCost = 1e13
	MESSAGE_LENGTH = PAYLOAD * IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNEL

	if not os.path.exists(result_file):
		with open(result_file, 'w+') as f_s:
			f_s.write('model:,%s\n' % model_dir)
			f_s.write('sample:,%s\n' % sample_file)
			f_s.write('stego:,%s\n' % stego_dir)
			f_s.write('item,image,success,sigma,loss0,loss,diff0,diff,stc,time\n')
	count = 0
	while (count < len(fileList)):
		# [2021.01.27]if the stego exists and cannot deceive the CNN, reproduce it again.
		if (check_stego == 1) & (os.path.exists(stego_dir + '/' + fileList[count])):
			coefS = read_single_ppm(stego_dir, fileList[count])
			data_x[0, :, :, :] = coefS.astype(np.float32)
			tempA, temp0, tempS = sess.run([accuracy, loss, s], feed_dict={x: data_x, y: data_y, is_train: False})
			if temp0 > 0.7:
				count += 1
				continue

		coefC = read_single_ppm(cover_dir, fileList[count])
		rhoCPV = read_cpv_cost(cost_dir, fileList[count], IMAGE_SIZE)
		stc_code = IS_STC

		print('count: %d, img: %s, net: XuNet, cost: %s, payload: %.2f' % (count, fileList[count], COST_FUNC, PAYLOAD))

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
				rhoCPV_c[
					randMatrix < SIGMA] = WetCost  # (SIGMA) DCT is set as wetCost; (1-SIGMA) DCT for the first embedding
				rhoCPV_n[:, :, idx_cpv] = rhoCPV_c

			if (IS_STC == 1) & (stc_code == 1):
				m_block, stc_code = EmbeddingSTC_CPV(coefC, rhoCPV_n, PAYLOAD * (1 - SIGMA))
			else:
				m_block = embed_cpv(coefC, rhoCPV_n, round(MESSAGE_LENGTH * (1 - SIGMA)))

			coefS += m_block

			data_x[0, :, :, :] = coefS.astype(np.float32)
			tempA, temp0, tempS = sess.run([accuracy, loss, s], feed_dict={x: data_x, y: data_y, is_train: False})
			tempS = tempS.reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL)

			if SIGMA == 0:
				stego_0 = coefS.copy()
				loss_0 = temp0
				dif_0 = np.sum(coefC != coefS)
				dif_cs = dif_0
			else:
				rhoCPV_n = adjust_cpv_cost(rhoCPV, tempS, 1 / ALPHA, 0)  # 1/ALPHA: the function is implemented for ITE.
				for idx_cpv in range(26):
					rhoCPV_c = rhoCPV_n[:, :, idx_cpv]
					rhoCPV_c[
						randMatrix >= SIGMA] = WetCost  # (SIGMA) DCT is set as wetCost; (1-SIGMA) DCT for the first embedding
					rhoCPV_n[:, :, idx_cpv] = rhoCPV_c
				if (IS_STC == 1) & (stc_code == 1):
					m_block, stc_code = EmbeddingSTC_CPV(coefC, rhoCPV_n, PAYLOAD * SIGMA)
				else:
					if IS_STC == 1:
						m_block, stc_code = EmbeddingSTC_CPV(coefC, rhoCPV_n, PAYLOAD * SIGMA * 1.05)
					else:
						m_block = embed_cpv(coefC, rhoCPV_n, round(MESSAGE_LENGTH * SIGMA))
				coefS += m_block

				data_x[0, :, :, :] = coefS.astype(np.float32)
				tempA, temp0, tempS = sess.run([accuracy, loss, s], feed_dict={x: data_x, y: data_y, is_train: False})
				dif_cs = np.sum(coefC != coefS)

			print('STC: %d, sigma: %.2f, acc: %.4f, loss: %.4f, dif: %d' % (stc_code, SIGMA, tempA, temp0, dif_cs))

			if temp0 > 0.7:
				break

		time_s = time.time()-time_0
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
		#sio.savemat(pathO+'/'+fileList[count],{'spatC':spatC,'spatS':spatS})
		count = count+1

	print("final result")
	print('Acc: %.4f, loss: %.4f\n' % (np.mean(Accuracy), np.mean(Loss)))
