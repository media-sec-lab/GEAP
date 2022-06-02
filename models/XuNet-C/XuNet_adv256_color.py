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
COST_FUNC = 'HILL'
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


def EmbeddingSimulator(rhoP, rhoM, m):
	n = rhoP.size
	Lambda = calc_lambda(rhoP, rhoM, m, n)

	zp = np.exp(-Lambda*rhoP)
	zm = np.exp(-Lambda*rhoM)
	z0 = 1+zp+zm
	pChangeP1 = zp/z0
	pChangeM1 = zm/z0
	
	#if fixEmbeddingChanges == 1:#RandStream.setGlobalStream(RandStream('mt19937ar','seed',139187))
	
	#else:#RandStream.setGlobalStream(RandStream('mt19937ar','Seed',sum(100*clock)))
	
	randChange = np.random.rand(rhoP.shape[0], rhoP.shape[1])
	modification = np.zeros([rhoP.shape[0], rhoP.shape[1]])
	modification[randChange < pChangeP1] = 1
	modification[randChange >= 1-pChangeM1] = -1
	return modification
	

# -, +, the bits that should be embedded, number of image
def calc_lambda(rhoP, rhoM, message_length, n):
	l3 = 1e+3
	m3 = message_length+1
	iterations = 0
	while m3 > message_length:
		l3 = l3*2
		zp = np.exp(-l3*rhoP)
		zm = np.exp(-l3*rhoM)
		z0 = 1+zp+zm
		pP1 = zp/z0
		pM1 = zm/z0
		m3 = ternary_entropyf(pP1, pM1)
		iterations = iterations+1
		if iterations > 10:
			Lambda = l3
			return Lambda

	l1 = 0
	m1 = n
	Lambda = 0

	alpha = float(message_length)/n
	while (float(m1-m3)/n > alpha/1000.0) and (iterations < 30):
		Lambda = l1+(l3-l1)/2.0
		zp = np.exp(-Lambda*rhoP)
		zm = np.exp(-Lambda*rhoM)
		z0 = 1+zp+zm
		pP1 = zp/z0
		pM1 = zm/z0
		m2 = ternary_entropyf(pP1, pM1)
		if m2 < message_length:
			l3 = Lambda
			m3 = m2
		else:
			l1 = Lambda
			m1 = m2
		iterations = iterations+1
	return Lambda


def ternary_entropyf(pP1, pM1):
	p0 = 1-pP1-pM1
	p0[p0 == 0] = 1e-10
	pP1[pP1 == 0] = 1e-10
	pM1[pM1 == 0] = 1e-10

	P = np.array([p0, pP1, pM1]).flatten()
	H = - (P * np.log2(P))
	# H((P < np.spacing(1)) | (P > 1 - np.spacing(1))) = 0
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
	ALPHA = 2
	# SUBIMAGE_SIZE = IMAGE_SIZE/BLOCK_STRIDE
	WetCost = 1e13

	with open(result_file, 'w+') as f_s:
		f_s.write('model:,%s\n' % model_dir)
		f_s.write('sample:,%s\n' % sample_file)
		f_s.write('stego:,%s\n' % stego_dir)
		f_s.write('item,image,success,sigma,loss0,loss,diff0,diff,stc,time\n')
	count = 0
	while (count < len(fileList)):
		#dataC = h5py.File(pathI+fileList[count].replace('.pgm', '.mat'))
		# dataC = sio.loadmat(cover_dir+'/'+fileList[count])
		# coefC = dataC['coef']
		# rhoP1 = dataC['rhoP1']
		# rhoM1 = dataC['rhoM1']
		coefC = read_single_ppm(cover_dir, fileList[count])
		rhoP1, rhoM1 = read_cost(cost_dir, fileList[count])
		stc_code = IS_STC

		print('count: %d' % (count))

		time_0 = time.time()
		LossPerImage = np.array([])

		sigma_val = 0
		stego_0 = coefC.copy()
		loss_0 = 0
		dif_0 = 0
		dif_cs = 0
		coefS = coefC.copy()
		temp0 = 0
		tempA = 0
		np.random.seed(5678)
		randMatrix = np.random.rand(coefC.shape[0], coefC.shape[1])
		for SIGMA in np.arange(0, 1, 0.1):
			coefS = coefC.copy()
			rhoP1N = rhoP1.copy()
			rhoM1N = rhoM1.copy()
			for idx_channel in range(NUM_CHANNEL):
				cover_c = coefC[:, :, idx_channel]
				stc_code = IS_STC
				sigma_val = SIGMA
				#randMatrix = np.random.rand( coefS.shape[0],coefS.shape[1] )
				rhoP1N_c = rhoP1N[:, :, idx_channel]
				rhoM1N_c = rhoM1N[:, :, idx_channel]
				stego_c = coefS[:, :, idx_channel]

				rhoP1N_c[randMatrix < SIGMA] = WetCost  #(SIGMA) DCT is set as wetCost; (1-SIGMA) DCT for the first embedding
				rhoM1N_c[randMatrix < SIGMA] = WetCost
				if IS_STC == 1:
					m_block, stc_code = EmbeddingSTC(cover_c, rhoP1N_c, rhoM1N_c, PAYLOAD*(1-SIGMA), STC_H)
				else:
					m_block = EmbeddingSimulator(rhoP1N_c, rhoM1N_c, round(PAYLOAD*IMAGE_SIZE*IMAGE_SIZE*(1-SIGMA)))

				coefS[:, :, idx_channel] = stego_c+m_block

			data_x[0, :, :, :] = coefS.astype(np.float32)
			tempA, temp0, tempS = sess.run([accuracy, loss, s], feed_dict={x: data_x, y: data_y, is_train: False})
			tempS = tempS.reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL)

			if SIGMA == 0:
				stego_0 = coefS.copy()
				loss_0 = temp0
				dif_0 = np.sum(coefC != coefS)
				dif_cs = dif_0
			else:
				rhoP1N = rhoP1.copy()
				rhoM1N = rhoM1.copy()
				rhoP1N[tempS == 1] = rhoP1N[tempS == 1]/ALPHA
				rhoM1N[tempS == 1] = rhoM1N[tempS == 1]*ALPHA
				rhoP1N[tempS == -1] = rhoP1N[tempS == -1]*ALPHA
				rhoM1N[tempS == -1] = rhoM1N[tempS == -1]/ALPHA
				for idx_channel in range(NUM_CHANNEL):
					cover_c = coefC[:, :, idx_channel]
					rhoP1N_c = rhoP1N[:, :, idx_channel]
					rhoM1N_c = rhoM1N[:, :, idx_channel]
					rhoP1N_c[randMatrix >= SIGMA] = WetCost  #(1-SIGMA) DCT is set as wetCost; (SIGMA) DCT for the second embedding
					rhoM1N_c[randMatrix >= SIGMA] = WetCost
					if (IS_STC == 1) & (stc_code == 1):
						m_block, stc_code = EmbeddingSTC(cover_c, rhoP1N_c, rhoM1N_c, PAYLOAD * SIGMA, STC_H)
					else:
						if IS_STC == 1:
							m_block = EmbeddingSimulator(rhoP1N_c, rhoM1N_c, round(1.05 * PAYLOAD * IMAGE_SIZE * IMAGE_SIZE * SIGMA))
						else:
							m_block = EmbeddingSimulator(rhoP1N_c, rhoM1N_c, round(PAYLOAD * IMAGE_SIZE * IMAGE_SIZE * SIGMA))
					coefS[:, :, idx_channel] = coefS[:, :, idx_channel]+m_block

				data_x[0, :, :, :] = coefS.astype(np.float32)
				tempA, temp0, tempS = sess.run([accuracy, loss, s], feed_dict={x: data_x, y: data_y, is_train: False})
				dif_cs = np.sum(coefC != coefS)

			print('STC: %d, sigma: %.2f, acc: %.4f, loss: %.4f, dif: %d' % (stc_code, SIGMA, tempA, temp0, dif_cs))

			LossPerImage = np.insert(LossPerImage, 0, temp0)
			if temp0 > 0.7:
				break

		time_s = time.time()-time_0
		if temp0 > 0.7:  #Adversarial examples can fool detector
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

		save_ppm(coefS.astype(np.uint8), stego_dir, fileList[count])
		print('acc: %.4f, loss: %.4f, dif: %d, time: %.4fs' % (tempA, temp0, dif_cs, time_s))
		Accuracy = np.insert(Accuracy, 0, tempA)
		Loss = np.insert(Loss, 0, temp0)
		#sio.savemat(pathO+'/'+fileList[count],{'spatC':spatC,'spatS':spatS})
		count = count+1

	print("final result")
	print('Acc: %.4f, loss: %.4f\n' % (np.mean(Accuracy), np.mean(Loss)))
