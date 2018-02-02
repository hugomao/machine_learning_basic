import pandas as pd
import numpy as np

def pca(x):
	num_data,num_var = x.shape
	#centralize x first
	mean_x = x.mean(axis = 0)
	x = x-mean_x
	#when numebr of feature is smaller than row count
	if num_data >= num_var:
		U, S, V = np.linalg.svd(x)
	else:
		#http://www.cnblogs.com/hanahimi/p/4312175.html
		#紧致技巧
		M = np.dot(x,x.T)
		e,ev = np.linalg.eigh(M)
		tmp = np.dot(x.T,ev)
		V = tmp[::-1]
		S = np.sqrt(e)[::-1]
		for i in range(V.shape[1]):
			V[:,i] / = S
	return V,S,mean_x


#https://zhuanlan.zhihu.com/p/21583787
#explanation of kernel pca: http://blog.csdn.net/lanchunhui/article/details/50492482
def kernel_pca(x):





