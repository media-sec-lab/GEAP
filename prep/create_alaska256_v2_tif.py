import os
import numpy as np
f1 = open('/data1/dataset/ALASKA256_v2/config/alaska256_v2_val.txt', 'w+')
f2 = open('/data1/dataset/ALASKA256_v2/config/alaska256_v2_tst.txt', 'w+')
f3 = open('/data1/dataset/ALASKA256_v2/config/alaska256_v2_trn.txt', 'w+')

path1 = '/data1/dataset/ALASKA256_v2/TIFF'
#path2 = '/data/shixiaoyu/imgs_tsq/s_40/'
for (dirpath,dirnames,filenames) in os.walk(path1):
	L = filenames
#L=[a for a in range(1,10001)]
np.random.seed(1234)
np.random.shuffle(L)


def write(file, a, b):
	for i in range(a-1, b):
		file.write(L[i][:-4]+'.mat\n')
#		file.write(path2+str(L[i])+'.pgm'+'\n')


write(f1, 1, 6000)  # val, 6000, 1/10 of training + validation.
f1.close()

write(f2, 6001, 26000)  # tst, 20000
f2.close()

write(f3, 26001, 80000)  # trn, 54000, 9/10 of training + validation.
f3.close()
