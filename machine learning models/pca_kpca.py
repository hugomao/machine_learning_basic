import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

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
			V[:,i] /= S
	return V,S,mean_x


#https://zhuanlan.zhihu.com/p/21583787
#explanation of kernel pca: http://blog.csdn.net/lanchunhui/article/details/50492482
#apply rbf kernel here
def kernel_pca(x,gamma,k):
    #calculate pairwise distance
    sq_dists = pdist(x, metric='sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    #calculate kernel value
    K = np.exp(-gamma*mat_sq_dists)
    
    N = X.shape[0]
    one_N = np.ones((N, N))/N
    #centralize kernel
    K = K-one_N.dot(K)-K.dot(one_N)+one_N.dot(K).dot(one_N)
    
    #calculate e/ev for kernel
    e,ev = np.linalg.eigh(K)
    alphas = np.stack((ev[:,-i] for i in range(1,k+1)),axis = 1)
    lambdas = [e[-i] for i in range(1, k+1)]
    return alphas, lambdas

def kernel_pca_proj(x_new,x,alphas,lambdas,gamma):
    K_new = np.exp(-gamma*np.sum((x-x_new)**2, 1))
    return K_new.dot(alphas / np.sqrt(lambdas))  #normalize alpha





