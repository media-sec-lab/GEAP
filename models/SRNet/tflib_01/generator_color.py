import numpy as np
import tensorflow as tf
# from scipy import misc, io
from scipy import misc, io, ndimage, fftpack
from glob import glob
import math
import random
# from itertools import izip
from random import random as rand
from random import shuffle
import scipy.io as sio
import os

def gen_flip_and_rot(cover_list, stego_list, quant_table=None, idct_kernel=None,
                     thread_idx=0, n_threads=10):

    nb_data = len(cover_list)
    # assert len(stego_list) != 0, "the stego directory '%s' is empty" % stego_dir
    # assert nb_data != 0, "the cover directory '%s' is empty" % cover_dir
    assert len(stego_list) != 0, "the stego list is empty"
    assert nb_data != 0, "the cover list is empty"
    assert len(stego_list) == nb_data, "the cover directory and " + \
                                      "the stego directory don't " + \
                                      "have the same number of files " + \
                                      "respectively %d and %d" % (nb_data, + \
                                      len(stego_list))

    load_ppm = 2
    if cover_list[0].endswith('.ppm'):
        load_ppm = 0
    elif cover_list[0].endswith('.mat'):
        load_ppm = 1

    if load_ppm == 0:
        img = ndimage.imread(cover_list[0])
        img_shape = img.shape
        batch = np.empty((2, img_shape[0], img_shape[1], img_shape[2]), dtype='float32')
    elif load_ppm == 1:
        if quant_table is None:
            dataC = sio.loadmat(cover_list[0])
            img = dataC['coefC']
        else:  # Jpeg
            dataC = sio.loadmat(cover_list[0])
            img = dataC['coef']

        img_shape = img.shape
        batch = np.empty((2, img_shape[0], img_shape[1], img_shape[2]), dtype='float32')
    else:
        img = misc.imread(cover_list[0])
        img_shape = img.shape
        batch = np.empty((2, img_shape[0], img_shape[1], img_shape[2]), dtype='uint8')

    tf.reset_default_graph()

    iterable = zip(cover_list, stego_list)
    while True:
        shuffle(iterable)
        for cover_path, stego_path in iterable:
            if load_ppm == 0:
                batch[0, :, :, :] = ndimage.imread(cover_path)
                batch[1, :, :, :] = ndimage.imread(stego_path)
            elif load_ppm == 1:
                if quant_table is None:
                    # dataC = sio.loadmat(cover_path)
                    dataC = sio.loadmat(stego_path)
                    imgC = dataC['coefC']
                    batch[0, :, :, :] = imgC.astype(np.float32)
                    imgS = dataC['coefS']
                    batch[1, :, :, :] = imgS.astype(np.float32)
                else:  # Jpeg
                    dataC = sio.loadmat(cover_path)
                    imgC = dataC['coef']
                    dctC = imgC.astype(np.float32)
                    pixC = dct_pixel(dctC, quant_table)
                    batch[0, :, :, :] = pixC  #imgC.astype(np.float32)
                    dataS = sio.loadmat(stego_path)
                    imgS = dataS['coef']
                    dctS = imgS.astype(np.float32)
                    pixS = dct_pixel(dctS, quant_table)
                    batch[1, :, :, :] = pixS
            else:
                batch[0, :, :, :] = misc.imread(cover_path)
                batch[1, :, :, :] = misc.imread(stego_path)

            rot = random.randint(0, 3)
            if rand() < 0.5:
                yield [np.rot90(batch, rot, axes=[1, 2]), np.array([0, 1], dtype='uint8')]
            else:
                yield [np.flip(np.rot90(batch, rot, axes=[1, 2]), axis=2), np.array([0, 1], dtype='uint8')]


def gen_valid(cover_list, stego_list, quant_table=None, idct_kernel=None,
                     thread_idx=0, n_threads=10):

    nb_data = len(cover_list)
    # assert len(stego_list) != 0, "the stego directory '%s' is empty" % stego_dir
    # assert nb_data != 0, "the cover directory '%s' is empty" % cover_dir
    assert len(stego_list) != 0, "the stego list is empty"
    assert nb_data != 0, "the cover list is empty"
    assert len(stego_list) == nb_data, "the cover directory and " + \
                                      "the stego directory don't " + \
                                      "have the same number of files " + \
                                      "respectively %d and %d" % (nb_data, \
                                      len(stego_list))
    load_ppm = 2
    if cover_list[0].endswith('.ppm'):
        load_ppm = 0
    elif cover_list[0].endswith('.mat'):
        load_ppm = 1

    if load_ppm == 0:
        img = ndimage.imread(cover_list[0])
        img_shape = img.shape
        batch = np.empty((2, img_shape[0], img_shape[1], img_shape[2]), dtype='float32')
    elif load_ppm == 1:
        if quant_table is None:
            dataC = sio.loadmat(cover_list[0])
            img = dataC['coefC']
        else:
            dataC = sio.loadmat(cover_list[0])
            img = dataC['coef']
        img_shape = img.shape
        batch = np.empty((2, img_shape[0], img_shape[1], img_shape[2]), dtype='float32')
    else:
        img = misc.imread(cover_list[0])
        img_shape = img.shape
        batch = np.empty((2, img_shape[0], img_shape[1], img_shape[2]), dtype='uint8')
    img_shape = img.shape

    labels = np.array([0, 1], dtype='uint8')
    while True:
        for cover_path, stego_path in zip(cover_list, stego_list):
            if load_ppm == 0:
                batch[0, :, :, :] = ndimage.imread(cover_path)
                batch[1, :, :, :] = ndimage.imread(stego_path)
            elif load_ppm == 1:
                if quant_table is None:
                    # dataC = sio.loadmat(cover_path)
                    dataC = sio.loadmat(stego_path)
                    imgC = dataC['coefC']
                    batch[0, :, :, :] = imgC.astype(np.float32)
                    imgS = dataC['coefS']
                    batch[1, :, :, :] = imgS.astype(np.float32)
                else:  # Jpeg
                    dataC = sio.loadmat(cover_path)
                    imgC = dataC['coef']
                    dctC = imgC.astype(np.float32)
                    pixC = dct_pixel(dctC, quant_table)
                    batch[0, :, :, :] = pixC  #imgC.astype(np.float32)
                    dataS = sio.loadmat(stego_path)
                    imgS = dataS['coef']
                    dctS = imgS.astype(np.float32)
                    pixS = dct_pixel(dctS, quant_table)
                    batch[1, :, :, :] = pixS
            else:
                batch[0, :, :, :] = misc.imread(cover_path)
                batch[1, :, :, :] = misc.imread(stego_path)

            yield [batch, labels]


def gen_tag(cover_list, stego_list, cover_tag_dir, stego_tag_dir, quant_table=None, idct_kernel=None,
                     thread_idx=0, n_threads=10):

    nb_data = len(cover_list)
    # assert len(stego_list) != 0, "the stego directory '%s' is empty" % stego_dir
    # assert nb_data != 0, "the cover directory '%s' is empty" % cover_dir
    assert len(stego_list) != 0, "the stego list is empty"
    assert nb_data != 0, "the cover list is empty"
    assert len(stego_list) == nb_data, "the cover directory and " + \
                                      "the stego directory don't " + \
                                      "have the same number of files " + \
                                      "respectively %d and %d" % (nb_data, \
                                      len(stego_list))
    load_pgm = 2
    if cover_list[0].endswith('.pgm'):
        load_pgm = 0
    elif cover_list[0].endswith('.mat'):
        load_pgm = 1

    if load_pgm == 0:
        img = ndimage.imread(cover_list[0])
        img_shape = img.shape
        batch = np.empty((2, img_shape[0], img_shape[1], 1), dtype='float32')
    elif load_pgm == 1:
        if quant_table is None:
            dataC = sio.loadmat(cover_list[0])
            img = dataC['coefC']
        else:
            dataC = sio.loadmat(cover_list[0])
            img = dataC['coef']
        img_shape = img.shape
        batch = np.empty((2, img_shape[0], img_shape[1], 1), dtype='float32')
    else:
        img = misc.imread(cover_list[0])
        img_shape = img.shape
        batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='uint8')
    # img_shape = img.shape

    labels = np.array([0, 1], dtype='uint8')
    while True:
        f_id = open('/data1/dataset/BOSS256_20k/Q75/SoftTag/file_list0.csv', 'w+')
        f_id.write('file name\n')
        for cover_path, stego_path in zip(cover_list, stego_list):
            if load_pgm == 0:
                batch[0, :, :, 0] = ndimage.imread(cover_path)
                batch[1, :, :, 0] = ndimage.imread(stego_path)
            elif load_pgm == 1:
                if quant_table is None:
                    dataC = sio.loadmat(cover_path)
                    imgC = dataC['coefC']
                    batch[0, :, :, 0] = imgC.astype(np.float32)
                    imgS = dataC['coefS']
                    batch[1, :, :, 0] = imgS.astype(np.float32)
                else:  # Jpeg
                    dataC = sio.loadmat(cover_path)
                    imgC = dataC['coef']
                    dctC = imgC.astype(np.float32)
                    pixC = dct_pixel(dctC, quant_table)
                    batch[0, :, :, 0] = pixC  #imgC.astype(np.float32)
                    dataS = sio.loadmat(stego_path)
                    imgS = dataS['coef']
                    dctS = imgS.astype(np.float32)
                    pixS = dct_pixel(dctS, quant_table)
                    batch[1, :, :, 0] = pixS
            else:
                batch[0, :, :, 0] = misc.imread(cover_path)
                batch[1, :, :, 0] = misc.imread(stego_path)

            d_name, f_name = os.path.split(cover_path)
            f_id.write(f_name + '\n')
            c_file = cover_tag_dir+'/'+f_name
            d_name, f_name = os.path.split(stego_path)
            s_file = stego_tag_dir+'/'+f_name
            names = np.array([c_file, s_file], dtype=np.str)
            yield [batch, labels, names.T]
        f_id.close()


def gen_type2(cover, stego, quant_table=None, idct_kernel=None,
                     thread_idx=0, n_threads=10):

    load_pgm = 2
    if cover.endswith('.pgm'):
        load_pgm = 0
    elif cover.endswith('.mat'):
        load_pgm = 1

    if load_pgm:
        img = ndimage.imread(cover)
        img_shape = img.shape
        batch = np.empty((2, img_shape[0], img_shape[1],1), dtype='float32')
    elif load_pgm == 1:
        if quant_table is None:
            dataC = sio.loadmat(cover)
            img = dataC['coefC']
        else:  # Jpeg
            dataC = sio.loadmat(cover)
            img = dataC['coef']
        img_shape = img.shape
        batch = np.empty((2, img_shape[0], img_shape[1], 1), dtype='float32')
    else:
        img = misc.imread(cover)
        img_shape = img.shape
        batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='uint8')

    labels = np.array([0, 1], dtype='uint8')
    while True:
        if load_pgm == 0:
            batch[0, :, :, 0] = ndimage.imread(cover)
            batch[1, :, :, 0] = ndimage.imread(stego)
        elif load_pgm == 1:
            if quant_table is None:
                dataC = sio.loadmat(cover)
                imgC = dataC['coefC']
                batch[0, :, :, 0] = imgC.astype(np.float32)
                imgS = dataC['coefS']
                batch[1, :, :, 0] = imgS.astype(np.float32)
            else:  # Jpeg
                dataC = sio.loadmat(cover)
                imgC = dataC['coef']
                dctC = imgC.astype(np.float32)
                pixC = dct_pixel(dctC, quant_table)
                batch[0, :, :, 0] = pixC  # imgC.astype(np.float32)
                dataS = sio.loadmat(stego)
                imgS = dataS['coef']
                dctS = imgS.astype(np.float32)
                pixS = dct_pixel(dctS, quant_table)
                batch[1, :, :, 0] = pixS
        else:
            batch[0, :, :, 0] = misc.imread(cover)
            batch[1, :, :, 0] = misc.imread(stego)

        if quant_table is not None:
            batch = dct_pixel(batch, quant_table, idct_kernel)

        yield [batch, labels]


def dct_pixel(dct_input, quant_table):
    idct_factor = np.array([[0.353553390593274, 0.353553390593274, 0.353553390593274,
                             0.353553390593274, 0.353553390593274, 0.353553390593274,
                             0.353553390593274, 0.353553390593274],
                            [0.490392640201615, 0.415734806151273, 0.277785116509801,
                             0.0975451610080642, -0.0975451610080641, -0.277785116509801,
                             -0.415734806151273, -0.490392640201615],
                            [0.461939766255643, 0.191341716182545, -0.191341716182545,
                             -0.461939766255643, -0.461939766255643, -0.191341716182545,
                             0.191341716182545, 0.461939766255643],
                            [0.415734806151273, -0.0975451610080641, -0.490392640201615,
                             -0.277785116509801, 0.277785116509801, 0.490392640201615,
                             0.0975451610080644, -0.415734806151273],
                            [0.353553390593274, -0.353553390593274, -0.353553390593274,
                             0.353553390593274, 0.353553390593274, -0.353553390593273,
                             -0.353553390593274, 0.353553390593273],
                            [0.277785116509801, -0.490392640201615, 0.0975451610080642,
                             0.415734806151273, -0.415734806151273, -0.0975451610080640,
                             0.490392640201615, -0.277785116509801],
                            [0.191341716182545, -0.461939766255643, 0.461939766255643,
                             -0.191341716182545, -0.191341716182545, 0.461939766255643,
                             -0.461939766255643, 0.191341716182545],
                            [0.0975451610080642, -0.277785116509801, 0.415734806151273,
                             -0.490392640201615, 0.490392640201615, -0.415734806151273,
                             0.277785116509801, -0.0975451610080643]], dtype=np.float32)

    # x_t = tf.multiply(dct_input, quant_table)
    x_t = np.multiply(dct_input, quant_table)
    x_shape = dct_input.shape
    outputs = dct_input.copy()
    for i in range(0, x_shape[0], 8):
        for j in range(0, x_shape[1], 8):
            y0 = x_t[i:i+8, j:j+8]
            y1 = np.dot(np.transpose(idct_factor), y0)
            y1 = np.dot(y1, idct_factor)+128
            outputs[i:i+8, j:j+8] = y1

    return outputs



