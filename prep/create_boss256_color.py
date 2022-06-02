import os
import numpy as np
f1 = open('/jbod/data/qinxinghong/BOSS256_C/config/boss256_color_val.txt', 'w+')
f2 = open('/jbod/data/qinxinghong/BOSS256_C/config/boss256_color_tst.txt', 'w+')
f3 = open('/jbod/data/qinxinghong/BOSS256_C/config/boss256_color_trn.txt', 'w+')

path1 = '/jbod/data/qinxinghong/BOSS256_C/BOSS_PPM256'
#path2 = '/data/shixiaoyu/imgs_tsq/s_40/'
for (dirpath,dirnames,filenames) in os.walk(path1):
	L = filenames
#L=[a for a in range(1,10001)]
np.random.seed(1234)
np.random.shuffle(L)


def write(file, a, b):
	for i in range(a-1, b):
		file.write(L[i]+'\n')
#		file.write(path2+str(L[i])+'.pgm'+'\n')


write(f1, 1, 1000)  # val, 1000
f1.close()

write(f2, 1001, 3000)  # tst, 2000
f2.close()

write(f3, 3001, 10000)  # trn, 7000
f3.close()
