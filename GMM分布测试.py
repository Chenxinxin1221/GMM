import numpy as np
import matplotlib.pyplot as plt
mean = np.array([3,10])
# 代表协方差cov(X,Y)=0,方差D(X)=D(Y)=1
cov = np.eye(2)    # np.eye(a)返回二维数组，对角线为1；a有几个数就有几个1
# print(cov)
num = 100
# np.random.multivariate_normal从多元正态分布中随机抽取样本
Y = np.random.multivariate_normal(mean, cov, num)    # 均值和协方差（必须有相同的长度）有几个数 得到的Y就有多少列（每列对应不同的均值）
print(Y)
# print(np.var(Y))
plt.scatter(Y[:,0], Y[:,1], c='r', marker='*')   # X轴代表均值为3，Y轴代表均值为10的点的分布
plt.show()
