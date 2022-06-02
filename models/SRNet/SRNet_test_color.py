# -*- coding: UTF-8 -*-
# Date: 2020.11.07
# 1.retrieve pre-defined parameters from the config file.
# 2.images' data is stored in mat files.
# ------------------------------
# change the learning rate bound from 400k to 200k
# related models are saved in '/home/liqs/recur/SRNet/models/SUniward04_200k/'
# 20181221 liqs
# 将max_iter=50w 改为 1w; save_interval=5k 改为 10；隐藏test代码
# 模型保存地址：‘/home/liqs/recur/SRNet/models/SUniward04_10k/’

import numpy as np
import scipy.io as sio
import os
import sys
import getopt
import ConfigParser as configparser
# import configparser

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3' # set a GPU (with GPU Number)
# home = os.path.expanduser("~")
# sys.path.append(home + '/recur/SRNet/tflib/')        # path for 'tflib' folder
sys.path.append('./tflib_01/')        # path for 'tflib' folder
from SRNet import *
from generator_color import *

# 875 iter/epoch
HOST_NAME = '172.31.224.46'
train_batch_size = 16
valid_batch_size = 16
test_batch_size = 16
max_iter = 500000
train_interval = 100
valid_interval = 5000
save_interval = 5000
num_runner_threads = 10
# for jpeg format samples.
# JPEG section
is_jpeg = 0
quant_factor = 0
quant_file = ''
dct2pixel_Kernel = None
quant_table = None

DIS_VER = 0
DB_SET = 'BOSS256_20k'
COST_FUNC = 'HILL'
PAYLOAD = 0.4
NUM_ITER = 500000

# Path section
cover_dir = ''
sample_file = './config/boss256_tst.txt'
log_dir = ''
result_file = ''

# Randomly chosen 4,000 images from BOSSbase and the entire BOWS2 dataset were used for training with 1,000 BOSSbase images set aside for validation.
# The remaining 5,000 BOSSbase images were used for testing.

# Cover and Stego directories for training and validation. For the spatial domain put cover and stego images in their 
# corresponding direcotries. For the JPEG domain, decompress images to the spatial domain without rounding to integers and 
# save them as '.mat' files with variable name "im". Put the '.mat' files in thier corresponding directoroies. Make sure 
# all mat files in the directories can be loaded in Python without any errors.

# Boss_256, Boss_256_suniward40, BOWS2_256, BOWS2_256_suniward40
config_file = './config/tst_SRNet_XuNet_HILL_BOSS256_p40.csv'

opts, args = getopt.getopt(sys.argv[1:], 'hc:t:', ['help', 'config=', 'host='])

for opt, val in opts:
    if opt == '-c':
        config_file = val
    if opt == '-t':
        HOST_NAME = val
    if opt in ('-h', '--help'):
        print('Steganalyze samples by using XuNet (with 6 CNN groups).')
        print('  -c: the config file.')
        sys.exit()

# # Read configs.
# config_raw = configparser.RawConfigParser()
# config_raw.read(config_file)

ste_ver_list = np.array([], dtype=np.int)
ste_path_list = np.array([], dtype=np.str)
with open(config_file, 'r') as f_id:
    lines = f_id.readlines()
    for s_line in lines:
        s_temp = s_line.split(',')
        key_t = s_temp[0]
        key_t = key_t.upper()
        if key_t == 'BATCH_SIZE':
            test_batch_size = int(s_temp[1])
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
        elif key_t == 'DISCRINIATOR':
            if len(s_temp) > 2 and len(s_temp[2]) > 3 and s_temp[2].strip() != HOST_NAME:
                continue
            log_dir = s_temp[1]
        elif key_t == 'NUM_ITERATION':
            NUM_ITER = int(s_temp[1])
        elif key_t == 'DIS_VER':
            DIS_VER = int(s_temp[1])
        elif key_t == 'STEGO':
            if len(s_temp) > 2 and len(s_temp[3]) > 3 and s_temp[3].strip() != HOST_NAME:
                continue
            ste_ver_list = np.append(ste_ver_list, int(s_temp[1]))
            ste_path_list = np.append(ste_path_list, s_temp[2])
        elif key_t == 'IS_JPEG':
            is_jpeg = int(s_temp[1])
        elif key_t == 'QUANT_FACTOR':
            quant_factor = float(s_temp[1])
        elif key_t == 'QUANT_FILE':
            quant_file = s_temp[1]

if is_jpeg == 1:
    quant_factor = config_raw.getfloat('Jpeg', 'quant_factor')
    quant_file = config_raw.get('Jpeg', 'quant_file')
    dataQ = sio.loadmat(quant_file)
    quant_table = dataQ['quant']

T = 1  # config_raw.getfloat('Basic', 'TEMPERATURE')

IS_DEBUG = 0

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices  # "7"
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(memory_gpu))
os.system('rm tmp')

path_r, file_r = os.path.split(result_file)
if not os.path.exists(path_r):
    os.mkdir(path_r)

if not os.path.exists(result_file):
    with open(result_file, 'w+') as f_r:
        f_r.write('dis ver,iter,cost,quality,payload,stego ver,loss,pe,model,stego,sample\n')

load_path = log_dir + '/Model_%d.ckpt' % NUM_ITER

optimizer = AdamaxOptimizer
boundaries = [400000]     # learning rate adjustment at iteration 400K
values = [0.001, 0.0001]  # learning rates

for STEGO_VER, sample_dir in zip(ste_ver_list, ste_path_list):
    sample_dir = sample_dir.strip()
    # with open(sample_file) as f:
    #     train_stego_names = f.readlines()
    #     train_cover_list = [cover_dir+'/'+a.strip() for a in train_stego_names]
    #     train_stego_list = [sample_dir+'/'+a.strip() for a in train_stego_names]
    with open(sample_file) as f:
        test_stego_names = f.readlines()
        test_cover_list = [cover_dir+'/'+a.strip() for a in test_stego_names]
        test_stego_list = [sample_dir+'/'+a.strip() for a in test_stego_names]

    print('Model: %s\nIteration: %d\nStego ver: %d\nStego: %s\n' \
          % (log_dir, DIS_VER, STEGO_VER, sample_dir))

    test_ds_size = len(test_cover_list) * 2
    # print('train_ds_size: %i' % train_ds_size)
    # print('valid_ds_size: %i' % valid_ds_size)
    print('test_ds_size: %i' % test_ds_size)

    if test_ds_size % test_batch_size != 0:
        raise ValueError("change batch size for testing.")

    # with open(validation_file) as f:
    #     valid_stego_names = f.readlines()
    #     valid_cover_list = [validation_cover+'/'+a.strip() for a in valid_stego_names]
    #     valid_stego_list = [validation_stego+'/'+a.strip() for a in valid_stego_names]

    # [2019.09.19]add parameters (quant_table and dct2pixel_Kernel) to handle inputting coef of jpeg.
    # train_gen = partial(gen_flip_and_rot,
    #                     train_cover_list, train_stego_list, quant_table)
    # valid_gen = partial(gen_valid,
    #                     valid_cover_list, valid_stego_list, quant_table)
    test_gen = partial(gen_valid,
                        test_cover_list, test_stego_list, quant_table)

    # Testing
    loss_0, acc_0 = test_dataset(SRNet, test_gen, test_batch_size, test_ds_size, load_path)
    with open(result_file, 'a+') as f_r:
        # f_r.write('dis ver,iter,cost,quality,payload,stego ver,loss,pe,model,stego,sample\n')
        f_r.write('%d,%d,%s,%.2f,%.2f,%d,%.4f,%.4f,%s,%s,%s\n' %
                  (DIS_VER, NUM_ITER, COST_FUNC, quant_factor, PAYLOAD, STEGO_VER
                   , loss_0, 1-acc_0, log_dir, sample_dir, sample_file))

    print('Loss=%.4f, acc=%.4f\n' % (loss_0, acc_0))

