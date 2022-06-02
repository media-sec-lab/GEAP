# -*- coding: UTF-8 -*-
# Date: 2019.09.16
# 1.retrieve pre-defined parameters from the config file.
# 2.images' data is stored in mat files.
# ------------------------------
# change the learning rate bound from 400k to 200k
# related models are saved in '/home/liqs/recur/SRNet/models/SUniward04_200k/'
# 20181221 liqs
# 将max_iter=50w 改为 1w; save_interval=5k 改为 10；隐藏test代码
# 模型保存地址：‘/home/liqs/recur/SRNet/models/SUniward04_10k/’

import scipy.io as sio
import os
import sys
import getopt
import ConfigParser as configparser
# import configparser
sys.path.append('./tflib_01/')        # path for 'tflib' folder
# from SRNet_color import *
from SRNet import *
from generator_color import *

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3' # set a GPU (with GPU Number)
# home = os.path.expanduser("~")
# sys.path.append(home + '/recur/SRNet/tflib/')        # path for 'tflib' folder

# 875 iter/epoch
train_batch_size = 16
valid_batch_size = 16
max_iter = 500000
train_interval=100
valid_interval=5000
save_interval=5000
num_runner_threads=10
# for jpeg format samples.
quant_factor = 0
quant_file = ''
dct2pixel_Kernel = None
quant_table = None
# Randomly chosen 4,000 images from BOSSbase and the entire BOWS2 dataset were used for training with 1,000 BOSSbase images set aside for validation.
# The remaining 5,000 BOSSbase images were used for testing.

# Cover and Stego directories for training and validation. For the spatial domain put cover and stego images in their 
# corresponding direcotries. For the JPEG domain, decompress images to the spatial domain without rounding to integers and 
# save them as '.mat' files with variable name "im". Put the '.mat' files in thier corresponding directoroies. Make sure 
# all mat files in the directories can be loaded in Python without any errors.

# Boss_256, Boss_256_suniward40, BOWS2_256, BOWS2_256_suniward40

config_file = './config/trn_SRNet_HILL_BOSS256_PPM_p40.cfg'

opts, args = getopt.getopt(sys.argv[1:], 'hc:', ['help', 'config='])
for opt, val in opts:
    if opt == '-c':
        config_file = val
    if opt in ('-h', '--help'):
        print('Steganalyze samples by using XuNet (with 6 CNN groups).')
        print('  -c: the config file.')
        sys.exit()

# Read configs.
config_raw = configparser.RawConfigParser()
config_raw.read(config_file)
visible_devices = config_raw.get('Environment', 'CUDA_VISIBLE_DEVICES')

train_batch_size = config_raw.getint('Basic', 'train_batch_size')  # 32
valid_batch_size = config_raw.getint('Basic', 'valid_batch_size')  # 40
NUM_ITER = config_raw.getint('Basic', 'NUM_ITER')
max_iter = config_raw.getint('Basic', 'MAX_ITER')  #10000
train_interval = config_raw.getint('Basic', 'train_interval')  # 100
valid_interval = config_raw.getint('Basic', 'valid_interval')  # 5000
save_interval = config_raw.getint('Basic', 'save_interval')  # 100
num_runner_threads = config_raw.getint('Basic', 'num_runner_threads')  # 10

T = config_raw.getfloat('Basic', 'TEMPERATURE')
LAMBDA = config_raw.getfloat('Basic', 'LAMBDA')  # 0.5

DIS_VER = config_raw.getint('Basic', 'DIS_VER')
DB_SET = config_raw.get('Basic', 'DB_SET')
COST_FUNC = config_raw.get('Basic', 'COST_FUNC')
PAYLOAD = config_raw.getfloat('Basic', 'PAYLOAD')
BETA = config_raw.getfloat('Basic', 'BETA')
# GAMMA = config_raw.getfloat('Basic', 'GAMMA')
# starter_learning_rate = config_raw.getfloat('Basic', 'starter_learning_rate')  # 0.001
# decay_steps = config_raw.getfloat('Basic', 'decay_steps')  # 5000
# decay_rate = config_raw.getfloat('Basic', 'decay_rate')  # 0.9

IS_DEBUG = config_raw.getfloat('Basic', 'IS_DEBUG')  # 0

# CC
model_cc = ''
num_item_cc = 0
num_boundary = 400000
if config_raw.has_option('Basic', 'NUM_BOUNDARY'):
    num_boundary = config_raw.getint('Basic', 'NUM_BOUNDARY')

# boundaries = [400000]     # learning rate adjustment at iteration 400K
if config_raw.has_option('CC', 'model'):
    model_cc = config_raw.get('CC', 'model')
    num_item_cc = config_raw.getint('CC', 'NUM_ITEM')
    num_boundary = config_raw.getint('CC', 'NUM_BOUNDARY')

# JPEG section
is_jpeg = config_raw.getint('Jpeg', 'is_jpeg')  # 1: jpeg samples.
if is_jpeg == 1:
    quant_factor = config_raw.getfloat('Jpeg', 'quant_factor')
    quant_file = config_raw.get('Jpeg', 'quant_file')
    dataQ = sio.loadmat(quant_file)
    quant_table = dataQ['quant']

# Path section
# model_dir = config_raw.get('Path', 'model_dir')
cover_dir = config_raw.get('Path', 'cover_dir')
sample_dir = config_raw.get('Path', 'sample_dir')
sample_file = config_raw.get('Path', 'sample_file')
cover_tag = config_raw.get('Path', 'cover_tag')
stego_tag = config_raw.get('Path', 'stego_tag')
log_dir = config_raw.get('Path', 'log_dir')
validation_cover = config_raw.get('Path', 'validation_cover')
validation_stego = config_raw.get('Path', 'validation_stego')
validation_file = config_raw.get('Path', 'validation_file')
# validation_result = config_raw.get('Path', 'validation_result')

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices  # "7"
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(memory_gpu))
os.system('rm tmp')

# with open("../datas/train.txt") as f:
#     train_stego_names = f.readlines()
#     train_stego_list = [a.strip() for a in train_stego_names]
# train_cover_list = [a.replace('_suniward40','') for a in train_stego_list]
with open(sample_file) as f:
    train_stego_names = f.readlines()
    train_cover_list = [cover_dir+'/'+a.strip() for a in train_stego_names]
    train_stego_list = [sample_dir+'/'+a.strip() for a in train_stego_names]

# with open("../datas/valid.txt") as f:
#     valid_stego_names = f.readlines()
#     valid_stego_list = [a.strip() for a in valid_stego_names]
# valid_cover_list = [a.replace('_suniward40','') for a in valid_stego_list]
with open(validation_file) as f:
    valid_stego_names = f.readlines()
    valid_cover_list = [validation_cover+'/'+a.strip() for a in valid_stego_names]
    valid_stego_list = [validation_stego+'/'+a.strip() for a in valid_stego_names]

# with open("../datas/test.txt") as f:
#     test_stego_names = f.readlines()
#     test_stego_list = [a.strip() for a in test_stego_names]
# test_cover_list = [a.replace('_suniward40','') for a in test_stego_list]

# [2019.09.19]add parameters (quant_table and dct2pixel_Kernel) to handle inputting coef of jpeg.
train_gen = partial(gen_flip_and_rot,
                    train_cover_list, train_stego_list, quant_table)
valid_gen = partial(gen_valid,
                    valid_cover_list, valid_stego_list, quant_table)

# LOG_DIR = '/home/liqs/recur/SRNet/models/SUniward04_10k/'  # path for a log direcotry
# load_path = LOG_DIR + 'Model_135000.ckpt'     # continue training from a specific checkpoint
load_path = None                              # training from scratch
if len(model_cc) > 3:
    load_path = model_cc + '/Model_%d.ckpt' % num_item_cc
elif NUM_ITER > 0:
    load_path = log_dir + '/Model_%d.ckpt' % NUM_ITER

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)

val_dir = log_dir + '/LogFile'
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

val_ret = val_dir + '/valid_acc.csv'
if not os.path.exists(val_ret):
    with open(val_ret, 'w+') as f_val:
        f_val.write('DB:,%s\nSamples:,%s\n' % (DB_SET, sample_dir))
        f_val.write('iter,loss,accuracy\n')

train_ds_size = len(train_cover_list) * 2
valid_ds_size = len(valid_cover_list) * 2
print('train_ds_size: %i' % train_ds_size)
print('valid_ds_size: %i' % valid_ds_size)

if valid_ds_size % valid_batch_size != 0:
    raise ValueError("change batch size for validation")
    
optimizer = AdamaxOptimizer
# boundaries = [400000]     # learning rate adjustment at iteration 400K
boundaries = [num_boundary]     # learning rate adjustment at iteration 400K
values = [0.001, 0.0001]  # learning rates

train(SRNet, train_gen, valid_gen , train_batch_size, valid_batch_size, valid_ds_size, \
      optimizer, boundaries, values, train_interval, valid_interval, max_iter,\
      save_interval, log_dir, num_runner_threads, load_path, T)
# T: for distillation.
# train(SRNet_color, train_gen, valid_gen , train_batch_size, valid_batch_size, valid_ds_size, \
#       optimizer, boundaries, values, train_interval, valid_interval, max_iter,\
#       save_interval, log_dir, num_runner_threads, load_path, T)

# Testing 
# Cover and Stego directories for testing
#TEST_COVER_DIR = '/data1/dataset/Boss/'
#TEST_STEGO_DIR = '/data1/dataset/BOSS_base_suniward40/'

#test_batch_size=40
#LOG_DIR = '/home/liqs/recur/SRNet/models/SUniward04_200k/' 
#LOAD_CKPT = LOG_DIR + 'Model_435000.ckpt'        # loading from a specific checkpoint

#test_gen = partial(gen_valid, \
#                    test_cover_list, test_stego_list)
#test_ds_size = len(test_cover_list) * 2
# #test_ds_size = len(glob(TEST_COVER_DIR + '/*')) * 2
#print 'test_ds_size: %i'%test_ds_size

#if test_ds_size % test_batch_size != 0:
#    raise ValueError("change batch size for testing!")

#test_dataset(SRNet, test_gen, test_batch_size, test_ds_size, LOAD_CKPT)




