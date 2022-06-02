# #2020.03.23
# #By: Qin Xinghong.
# #XuNet (with 6 groups for spatial images.)
# #-c: config file, in which there are discriminator, sample dir and sample file list.
# #Test	Model	DIS_VER	Iteration	STE_VER	Stego_dir

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
import sys
import getopt
import h5py
# import ConfigParser as configparser
# # import configparser
sys.path.append('./tflib/')        # path for 'tflib' folder
from generator import *

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

HOST_NAME = '172.31.224.44'
BATCH_COVER = 32
BATCH_SIZE = 2*BATCH_COVER
IMAGE_SIZE = 256
NUM_CHANNEL = 3
NUM_LABELS = 2
NUM_SHOWTRAIN = 200 #show result eveary epoch
NUM_SHOWTEST = 5000
#BN_DECAY = 0.95
#UPDATE_OPS_COLLECTION = 'Discriminative_update_ops
is_train = True
#NUM_ITER = 120000  # -i: iteration. The iteration of the discriminator.
DIS_VER = 0  # -d: discriminator version. 0: based version.
STEGO_VER = 0  # -s: stego version. 0: based version.
DB_SET = 'BOSS256_C'
COST_FUNC = 'HILL'  # -c: cost function. SUNIWARD is the default.
RES_TEST = True  # results of testing set. 0: results of training set; 1: results of testing set.
PAYLOAD = 0.4  # -p: payload
ALPHA = 2
BETA = 0.1
GAMMA = 0.01
STEP = 0.05
MARGINE = 0.01
sample_file = ''
cover_dir = '/data1/dataset/BOSS256_20k/SUNIWARD'
record_stego = 0
result_file = ''
record_dir = ''

config_file = './config_color/tst_XuNet_HILL_BOSS256_PPM_p40.csv'

opts, args = getopt.getopt(sys.argv[1:], 'hc:t:',	['help', 'config=', 'host='])
for opt, val in opts:
	if opt == '-c':
		config_file = val
	if opt == '-t':
		HOST_NAME = val
	if opt in ('-h', '--help'):
		print('Steganalyze samples by using XuNet (with 6 CNN groups).')
		print('  -c: config file, in which there are discriminator, sample dir and sample file list.')
		print('  -t: the host name or IP address of the server.')
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

# os.environ["CUDA_VISIBLE_DEVICES"] = visible_device  # '3'
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(memory_gpu))
os.system('rm tmp')

fileList = []
if len(sample_file) > 3:
	with open(sample_file, 'r') as f:
		lines = f.readlines()
		fileList = [a.strip() for a in lines]
else:
	for (dirpath, dirnames, filenames) in os.walk(cover_dir):
		fileList = filenames
NUM_SAMPLE = len(fileList)

x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_LABELS])
is_train = tf.placeholder(tf.bool, name='is_train')

hpf = np.zeros([5, 5, NUM_CHANNEL, 1], dtype=np.float32)
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
	# y_softmax = tf.nn.softmax(y_)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
correct_predictionCover, correct_predictionStego = tf.split(correct_prediction, 2, 0)
accuracyCover = tf.reduce_mean(tf.cast(correct_predictionCover, tf.float32))
accuracyStego = tf.reduce_mean(tf.cast(correct_predictionStego, tf.float32))

vars = tf.trainable_variables()
params = [v for v in vars if (v.name.startswith('Group'))]

# tf.summary.scalar('acc', accuracy)
loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_)
loss = tf.reduce_mean(loss_)
# tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 5000, 0.9, staircase=True)
opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss, var_list=params, global_step=global_step)

data_x = np.zeros([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
data_y = np.zeros([BATCH_SIZE, NUM_LABELS])
# for i in range(0, BATCH_COVER):
# 	data_y[i, 1] = 1
# for i in range(BATCH_COVER, BATCH_SIZE):
# 	data_y[i, 0] = 1
data_y[0:BATCH_COVER, 1] = 1
data_y[BATCH_COVER:, 0] = 1

# saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

for DIS_VER, model_dir, NUM_ITER, STEGO_VER, sample_dir \
		in zip(dis_ver_list, dis_model_list, dis_iter_list, ste_ver_list, ste_path_list):
	model_dir = model_dir.strip()
	sample_dir = sample_dir.strip()
	print('dis_ver: %d, model: %s\nsample_ver: %d, sample_dir: %s\n' % (DIS_VER, model_dir, STEGO_VER, sample_dir))
	with tf.Session(config=config) as sess:
		tf.global_variables_initializer().run()

		saver = tf.train.Saver()
		saver.restore(sess, model_dir+'/'+str(NUM_ITER)+'.ckpt')

		Loss = np.array([])
		Accuracy = np.array([])
		AccuracyCover = np.array([])
		AccuracyStego = np.array([])
		ArrayCount = [0]*BATCH_COVER

		count = 0
		while count < NUM_SAMPLE:
			for j in range(BATCH_COVER):
				if count >= NUM_SAMPLE:
					break

				ArrayCount[j] = count
				# dataC = sio.loadmat(sample_dir+'/'+fileList[count])
				# cover = dataC['coefC']
				# stego = dataC['coefS']
				cover, stego = read_ppm(cover_dir, sample_dir, fileList[count])
				data_x[j, :, :, :] = cover.astype(np.float32)
				data_x[j+BATCH_COVER, :, :, :] = stego.astype(np.float32)
				count = count+1
			tempA, tempB, tempC, temp0, temp1 = sess.run([accuracy, accuracyCover, accuracyStego, loss, loss_],
															feed_dict={x: data_x, y: data_y, is_train: False})
			print('%d-%d, count: %d, loss: %.4f, accuracy: %.4f' % (DIS_VER, STEGO_VER, count, temp0, tempA))
			# print(count,tempA,temp0)
			# print(1-tempC,1-tempA)
			Accuracy = np.insert(Accuracy, 0, tempA)
			AccuracyCover = np.insert(AccuracyCover, 0, tempB)
			AccuracyStego = np.insert(AccuracyStego, 0, tempC)
			Loss = np.insert(Loss, 0, temp0)
			#print(temp1)
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

		ErrCover = 1-np.mean(AccuracyCover)
		ErrStego = 1-np.mean(AccuracyStego)
		ErrRate = 1-np.mean(Accuracy)
		Loss_i = np.mean(Loss)
		print('%d-%d iter: %d, loss: %.4f, err cover: %.4f, err stego: %.4f, err rate: %.4f\n' %
				(DIS_VER, STEGO_VER, NUM_ITER, Loss_i, ErrCover, ErrStego, ErrRate))
		if len(result_file) > 3:
			if not os.path.isfile(result_file):
				with open(result_file, 'w+') as f_out:
					f_out.write(
						'discriminator,iter,cost,stego ver,payload,beta,gamma,loss,Pfa,Pmd,PE,sample list,model,sample dir\n')
					f_out.write('%d,%d,%s,%d,%.2f,%.2f,%.3f,%.4f,%.4f,%.4f,%.4f,%s,%s,%s\n' %
								(DIS_VER, NUM_ITER, COST_FUNC, STEGO_VER, PAYLOAD, BETA, GAMMA, Loss_i, ErrCover,
									ErrStego, ErrRate, sample_file, model_dir, sample_dir))
			else:
				with open(result_file, 'a+') as f_out:
					f_out.write('%d,%d,%s,%d,%.2f,%.2f,%.3f,%.4f,%.4f,%.4f,%.4f,%s,%s,%s\n' %
								(DIS_VER, NUM_ITER, COST_FUNC, STEGO_VER, PAYLOAD, BETA, GAMMA, Loss_i, ErrCover,
									ErrStego, ErrRate, sample_file, model_dir, sample_dir))

