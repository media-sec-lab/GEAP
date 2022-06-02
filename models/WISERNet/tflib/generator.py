
import sys
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os
import random
import math
import h5py
from scipy import ndimage, signal
import scipy.io as sio
import getopt
from PIL import Image, ImageDraw


def read_img(cover_dir, sample_dir, img_name):
	img_type = 0  # .mat
	if img_name.endswith('pgm') or img_name.endswith('PGM'):
		img_type = 1

	if img_type == 0:
		dataC = sio.loadmat(sample_dir + '/' + img_name)
		cover = dataC['coefC']
		stego = dataC['coefS']
	else:
		cover = ndimage.imread(cover_dir + '/' + img_name)
		stego = ndimage.imread(sample_dir + '/' + img_name)

	return cover, stego


def read_ppm(cover_dir, sample_dir, img_name):
	img_type = 0  # .mat
	if img_name.endswith('ppm') or img_name.endswith('PPM'):
		img_type = 1
	elif img_name.endswith('tif') or img_name.endswith('TIF'):
		img_type = 2

	if img_type == 0:
		dataC = sio.loadmat(sample_dir + '/' + img_name)
		cover = dataC['coefC']
		stego = dataC['coefS']
	else:
		cover = ndimage.imread(cover_dir + '/' + img_name)
		stego = ndimage.imread(sample_dir + '/' + img_name)

	return cover, stego


def read_single_ppm(sample_dir, img_name):
	img_type = 0  # .mat
	if img_name.endswith('ppm') or img_name.endswith('PPM'):
		img_type = 1
	elif img_name.endswith('tif') or img_name.endswith('TIF'):
		img_type = 2

	if img_type == 0:
		dataC = sio.loadmat(sample_dir + '/' + img_name)
		cover = dataC['coefC']
		# stego = dataC['coefS']
	else:
		cover = ndimage.imread(sample_dir + '/' + img_name)
		# stego = ndimage.imread(sample_dir + '/' + img_name)

	return cover


def read_cost(sample_dir, img_name):
	mat_name, ext_name = os.path.splitext(img_name)
	cost_mat = sio.loadmat(sample_dir+'/'+mat_name+'.mat')
	rho_p1 = cost_mat['rhoP1']
	rho_m1 = cost_mat['rhoM1']

	return rho_p1, rho_m1


def read_cpv_cost(sample_dir, img_name, img_size):
	mat_name, ext_name = os.path.splitext(img_name)
	cost_mat = sio.loadmat(sample_dir+'/'+mat_name+'.mat')
	rho_0 = cost_mat['rhoCPV']  # dimensions=27*...
	rho_27 = np.transpose(rho_0)
	rho_27 = rho_27.reshape(img_size, img_size, 27)
	rho_27 = np.transpose(rho_27, (1, 0, 2))
	rho_27 = rho_27.astype(np.float32)

	return rho_27


def save_ppm(img_data, save_dir, img_name):
	img = Image.fromarray(img_data.astype(np.uint8), mode='RGB')
	if img_name.endswith('mat') or img_name.endswith('MAT'):
		file_name, ext_name = os.path.splitext(img_name)
		img.save(os.path.join(save_dir, file_name+'.ppm'))
	else:
		img.save(os.path.join(save_dir, img_name))


def embed_cpv(X, rhoCPV, m):
	n = X.size
	Lambda = calc_lambda_cpv(rhoCPV, m, n)

	z_cpv = np.exp(-Lambda * rhoCPV)
	z0 = np.zeros([rhoCPV.shape[0], rhoCPV.shape[1]], dtype=np.float)
	for idx_z in range(27):
		z0 += z_cpv[:, :, idx_z]
	p_cpv = np.zeros(rhoCPV.shape, dtype=np.float)
	for idx_z in range(27):
		p_cpv[:, :, idx_z] = z_cpv[:, :, idx_z] / z0

	changes = np.array([[1, 1, 1], [-1, -1, -1], [1, 1, 0], [1, 0, 1], [1, 0, 0],
						[-1, -1, 0], [-1, 0, -1], [-1, 0, 0], [0, 1, 1], [0, 1, 0],
						[0, -1, -1], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 1, -1],
						[1, -1, 1], [1, -1, -1], [1, -1, 0], [1, 0, -1], [-1, 1, 1],
						[-1, 1, -1], [-1, 1, 0], [-1, -1, 1], [-1, 0, 1], [0, 1, -1],
						[0, -1, 1], [0, 0, 0]], dtype=np.int)
	randChange = np.random.rand(X.shape[0], X.shape[1])
	m_0 = np.zeros([X.shape[0], X.shape[1]], dtype=np.int32)
	m_1 = m_0.copy()
	m_2 = m_0.copy()
	p_tmp = np.zeros([X.shape[0], X.shape[1]], dtype=np.float)
	for idx_v in range(27):
		p_idx = p_tmp + p_cpv[:, :, idx_v]
		if changes[idx_v, 0] > 0:  # +1
			m_0[(randChange < p_idx) & (randChange >= p_tmp) & (X[:, :, 0] < 255)] = 1
		elif changes[idx_v, 0] < 0:  # -1
			m_0[(randChange < p_idx) & (randChange >= p_tmp) & (X[:, :, 0] > 0)] = -1

		if changes[idx_v, 1] > 0:  # +1
			m_1[(randChange < p_idx) & (randChange >= p_tmp) & (X[:, :, 1] < 255)] = 1
		elif changes[idx_v, 1] < 0:  # -1
			m_1[(randChange < p_idx) & (randChange >= p_tmp) & (X[:, :, 1] > 0)] = -1

		if changes[idx_v, 2] > 0:  # +1
			m_2[(randChange < p_idx) & (randChange >= p_tmp) & (X[:, :, 2] < 255)] = 1
		elif changes[idx_v, 2] < 0:  # -1
			m_2[(randChange < p_idx) & (randChange >= p_tmp) & (X[:, :, 2] > 0)] = -1
		p_tmp = p_idx   # += p_cpv[:, :, idx_v]

	modification = np.zeros([X.shape[0], X.shape[1], X.shape[2]], dtype=np.int32)
	modification[:, :, 0] = m_0
	modification[:, :, 1] = m_1
	modification[:, :, 2] = m_2
	return modification


def ternary_entropy_cpv(pCPV):
	pCPV[pCPV == 0] = 1e-10

	H = - (pCPV * np.log2(pCPV))
	# H((P < np.spacing(1)) | (P > 1 - np.spacing(1))) = 0
	H[(pCPV < 2.2204e-16) | (pCPV > 1 - 2.2204e-16)] = 0

	Ht = H.sum()
	return Ht


def calc_lambda_cpv(rhoCPV, message_length, n):
	l3 = 1e+3
	m3 = message_length+1
	iterations = 0
	while m3 > message_length:
		l3 = l3*2
		z_cpv = np.exp(-l3*rhoCPV)
		z0 = np.zeros([rhoCPV.shape[0], rhoCPV.shape[1]], dtype=np.float)
		for idx_z in range(27):
			z0 += z_cpv[:, :, idx_z]
		p_cpv = np.zeros(rhoCPV.shape, dtype=np.float)
		for idx_z in range(27):
			p_cpv[:, :, idx_z] = z_cpv[:, :, idx_z] / z0

		m3 = ternary_entropy_cpv(p_cpv)
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
		z_cpv = np.exp(-Lambda*rhoCPV)
		z0 = np.zeros([rhoCPV.shape[0], rhoCPV.shape[1]], dtype=np.float)
		for idx_z in range(27):
			z0 += z_cpv[:, :, idx_z]
		p_cpv = np.zeros(rhoCPV.shape, dtype=np.float)
		for idx_z in range(27):
			p_cpv[:, :, idx_z] = z_cpv[:, :, idx_z] / z0
		m2 = ternary_entropy_cpv(p_cpv)
		if m2 < message_length:
			l3 = Lambda
			m3 = m2
		else:
			l1 = Lambda
			m1 = m2
		iterations = iterations+1
	return Lambda


def adjust_cmd_cpv(rho_ori, img_cover, img_stego, cmd_factor):
	changes = np.array([[1, 1, 1], [-1, -1, -1], [1, 1, 0], [1, 0, 1], [1, 0, 0],
						[-1, -1, 0], [-1, 0, -1], [-1, 0, 0], [0, 1, 1], [0, 1, 0],
						[0, -1, -1], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 1, -1],
						[1, -1, 1], [1, -1, -1], [1, -1, 0], [1, 0, -1], [-1, 1, 1],
						[-1, 1, -1], [-1, 1, 0], [-1, -1, 1], [-1, 0, 1], [0, 1, -1],
						[0, -1, 1], [0, 0, 0]], dtype=np.int32)
	k_n = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
	rho_ret = rho_ori.copy()
	dif_emb = img_stego.astype(np.int32) - img_cover.astype(np.int32)
	for idx_c in range(26):
		chg_map = np.zeros([img_cover.shape[0], img_cover.shape[1]], dtype=np.float32)
		chg_map[(dif_emb[:, :, 0] == changes[idx_c, 0]) & (dif_emb[:, :, 1] == changes[idx_c, 1]) & \
										(dif_emb[:, :, 2] == changes[idx_c, 2])] = 1
		chg_map = signal.convolve2d(chg_map, k_n, boundary='symm', mode='same')
		rho_c = rho_ret[:, :, idx_c]
		rho_c[chg_map >= 1] *= cmd_factor
		rho_ret[:, :, idx_c] = rho_c

	return rho_ret


def adjust_cpv_cost(rho_ori, sign_grd, factor_adj, version=0):
	changes = np.array([[1, 1, 1], [-1, -1, -1], [1, 1, 0], [1, 0, 1], [1, 0, 0],
						[-1, -1, 0], [-1, 0, -1], [-1, 0, 0], [0, 1, 1], [0, 1, 0],
						[0, -1, -1], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 1, -1],
						[1, -1, 1], [1, -1, -1], [1, -1, 0], [1, 0, -1], [-1, 1, 1],
						[-1, 1, -1], [-1, 1, 0], [-1, -1, 1], [-1, 0, 1], [0, 1, -1],
						[0, -1, 1], [0, 0, 0]], dtype=np.int32)
	rho_ret = rho_ori.copy()
	if version == 1:  # component version.
		for idx_v in range(26):
			f_i = np.ones([sign_grd.shape[0], sign_grd.shape[1]], dtype=np.float32)
			if changes[idx_v, 0] > 0:  # R
				f_i[sign_grd[:, :, 0] > 0] *= factor_adj
				f_i[sign_grd[:, :, 0] < 0] /= factor_adj
			elif changes[idx_v, 0] < 0:
				f_i[sign_grd[:, :, 0] > 0] /= factor_adj
				f_i[sign_grd[:, :, 0] < 0] *= factor_adj
			if changes[idx_v, 1] > 0:  # G
				f_i[sign_grd[:, :, 1] > 0] *= factor_adj
				f_i[sign_grd[:, :, 1] < 0] /= factor_adj
			elif changes[idx_v, 1] < 0:
				f_i[sign_grd[:, :, 1] > 0] /= factor_adj
				f_i[sign_grd[:, :, 1] < 0] *= factor_adj
			if changes[idx_v, 2] > 0:  # B
				f_i[sign_grd[:, :, 2] > 0] *= factor_adj
				f_i[sign_grd[:, :, 2] < 0] /= factor_adj
			elif changes[idx_v, 2] < 0:
				f_i[sign_grd[:, :, 2] > 0] /= factor_adj
				f_i[sign_grd[:, :, 2] < 0] *= factor_adj
			rho_ret[:, :, idx_v] = rho_ret[:, :, idx_v] * f_i
	else:  # vector version.
		for idx_v in range(26):
			grd_l = np.zeros([sign_grd.shape[0], sign_grd.shape[1]], dtype=np.int)
			grd_l[(sign_grd[:, :, 0] == changes[idx_v, 0]) & (sign_grd[:, :, 1] == changes[idx_v, 1]) & \
									(sign_grd[:, :, 2] == changes[idx_v, 2])] = 1
			grd_l[(sign_grd[:, :, 0] == -changes[idx_v, 0]) & (sign_grd[:, :, 1] == -changes[idx_v, 1]) & \
									(sign_grd[:, :, 2] == -changes[idx_v, 2])] = -1
			rho_i = rho_ret[:, :, idx_v]
			rho_i[grd_l == 1] *= factor_adj
			rho_i[grd_l == -1] /= factor_adj
			rho_ret[:, :, idx_v] = rho_i

	return rho_ret


def adjust_adv_cost(rho_p, rho_m, sign_grd, factor_adj, version=0):
	rho_pn = rho_p.copy()
	rho_mn = rho_m.copy()

	if version == 2:
		f_p = np.ones((rho_p.shape[0], rho_p.shape[1]), dtype=np.float)
		f_m = np.ones((rho_p.shape[0], rho_p.shape[1]), dtype=np.float)
		for idx_c in range(rho_p.shape[2]):
			s_c = sign_grd[:, :, idx_c]
			f_p[s_c == 1] *= factor_adj
			f_m[s_c == 1] /= factor_adj
			f_p[s_c == -1] /= factor_adj
			f_m[s_c == -1] *= factor_adj
		f_p1 = np.ones(rho_p.shape, dtype=np.float)
		f_m1 = np.ones(rho_p.shape, dtype=np.float)
		for idx_c in range(rho_p.shape[2]):
			f_p1[:, :, idx_c] = f_p
			f_m1[:, :, idx_c] = f_m
		rho_pn *= f_p1
		rho_mn *= f_m1
	else:
		rho_pn[sign_grd == 1] *= factor_adj
		rho_mn[sign_grd == 1] /= factor_adj
		rho_pn[sign_grd == -1] /= factor_adj
		rho_mn[sign_grd == -1] *= factor_adj

	return rho_pn, rho_mn


def neighbor_change_color(cover, stego):
	h = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float)
	d = stego.astype(np.float32) - cover.astype(np.float32)
	c = np.zeros(cover.shape, dtype=np.float)
	for idx_v in range(cover.shape[2]):
		x = signal.convolve2d(d[:, :, idx_v], h, boundary='symm', mode='same')
		c[:, :, idx_v] = x
	return c


def adjust_cmd_color(rho_p, rho_m, img_cover, img_stego, cmd_factor):
	rho_pn = rho_p.copy()
	rho_mn = rho_m.copy()
	n_c = neighbor_change_color(img_stego, img_cover)
	rho_pn[n_c > 0] *= cmd_factor
	rho_mn[n_c < 0] *= cmd_factor

	return rho_pn, rho_mn


def cost_hill(X):
	hp = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32)
	r_1 = signal.convolve2d(X.astype(np.float32), hp, boundary='symm', mode='same')
	lp_1 = np.ones([3, 3], dtype=np.float32)/9
	r_2 = signal.convolve2d(np.abs(r_1), lp_1, boundary='symm', mode='same')
	rho_1 = 1/(r_2 + 1e-10)
	lp_2 = np.ones([15, 15], dtype=np.float32)/225
	rho_1 = signal.convolve2d(rho_1, lp_2, boundary='symm', mode='same')
	# rho_p = rho_1
	# rho_m = rho_1
	# rho_p[X == 255] = 1e10
	# rho_m[X == 0] = 1e10
	return rho_1
