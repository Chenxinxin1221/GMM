# import numpy as np
# a = np.arange(12).reshape(3, 4)
# print(len(a))   # 行数

# def test(a, *args, **kwargs):
#     print(a)
#     print(args)
#     print(kwargs)
# if __name__ == "__main__":
#     test(1, 2, 3, d='4', e=5)

# import numpy as np
# a = np.array([np.random.randint(0, 20, 5), np.random.randint(0, 20, 5)])
# print('原始数据\n', a)
# print('mean函数'.center(20, '*'))
# print('对所有数据计算均值\n', a.mean())
# print('axis=0，按行方向计算，即每列\n', a.mean(axis=0))  # 按行方向计算，即每列
# print('axis=1，按列方向计算，即每行\n', a.mean(axis=1))  # 按列方向计算，即每行
# print('average函数'.center(20, '*'))
# print('对所有数据计算\n', np.average(a))
# print('axis=0，按行方向计算，即每列\n', np.average(a, axis=0))  # 按行方向计算，即每列
# print('axis=1，按列方向计算，即每行\n', np.average(a, axis=1))  # 按列方向计算，即每行
# b = np.array([1, 2, 3, 4])
# wts = np.array([4, 3, 2, 1])
# print('不指定权重\n', np.average(b))
# print('指定权重\n', np.average(b, weights=wts))


# import numpy as np
# a = np.arange(6).reshape(-1,1)
# print(a)
# print(a[0])

# print(np.random.randint(0, 10, 5))  # 在0-9中随机选取5个整数（可重复）

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import multivariate_normal
# # x = np.linspace(0, 5, 10, endpoint=False
# x1 = np.arange(300).reshape(150,2)
# # x1是几维，后面的mean，cov就要对应是几维
# # 是这一组数共同在这个高斯分布中对应的概率
# # 数据要进行归一化处理！！！
# for i in range(x1.shape[1]):
#     sums = x1[:, i]
#     x1[:, i] = (x1[:, i] - np.min(x1[:, i])) / (np.max(x1[:, i]) - np.min(x1[:, i]))
# y = multivariate_normal.pdf(x1, mean=[2.5,2], cov=np.diag([0.5,0.5]))  # 得到的是x1的点在mean=2.5附近取值的可能性
# print(y)
# # plt.scatter(x1, y, s=20)
# # plt.show()

import numpy as np
# np.random.multivariate_normal
# 生成一个服从多元正态分布的数组　
# mean = (1, 2)
# cov = [[1, 0], [0, 1]]
# x = np.random.multivariate_normal(mean, cov, (2, 2), 'raise')   # 2x2x2
# print(x)

# scipy.stats.multivariate_normal
# 生成一个多元正态分布
# import numpy as np
# import scipy.stats as st
# import matplotlib.pylab as plt
# x, y = np.mgrid[-1:1:.01, -1:1:.01]
# pos = np.empty(x.shape + (2,))
# pos[:, :, 0] = x; pos[:, :, 1] = y
# rv = st.multivariate_normal([0, 0], [[1, 0], [0, 1]])   # 生成多元正态分布
# print(rv)       # <scipy.stats._multivariate.multivariate_normal> 只是生成了一个对象，并没有生成数组
# plt.contourf(x, y, rv.pdf(pos))
# plt.show()


# fig, axs = plt.subplots(2, 3)
# plt.scatter(np.arange(3), np.arange(3))  # 默认在最后一个图画
# ax1 = axs[1,1]
# ax1.plot(np.arange(4))
# plt.show()

# fig, axs = plt.subplots(2, 3)
# plt.scatter(np.arange(3), [6, 3, 4])  # 默认在最后一个图画
# ax1 = plt.subplot(2, 2, 1)
# ax2 = plt.subplot(2, 2, 2)
# # plt.plot(np.arange(10))
# ax = plt.gca()
# ax.plot(np.arange(10))
# ax3 = plt.subplot(2, 2, 3)
# ax4 = plt.subplot(2, 2, 4)
# plt.show()

# a = np.array([[12,132,43],[43,56,78]])
# print(a)
# print(a[0])
# print(a.shape)
# print(a.sum(axis=1))  # 计算每一行的总值
# print(a.sum(axis=1).reshape(-1, 1))  # 计算每一行的总值并转换为一列

# print(np.mean([1, 2, 12]))


# X = np.arange(10).reshape(5, 2)   # 5*2
# # print(X.shape)
# W = np.array([1,1,1,2,2])   # 必须是一维
# # print(W)
# # print(W.shape)
# # Mean = np.zeros((2, 1))
# # print(W)
# # print(Mean.shape)
# # Mean = np.average(X, axis=1, weights=W)
# print(np.average(X, axis=0, weights=W))

# data = np.arange(6).reshape((3,2))
# print(np.average(data, axis=1, weights=[1./4, 3./4]))
# print(np.average(data, axis=1, weights=[1./4, 3./4]).shape)

import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.Session()
x_val = tf.linspace(-10., 10., 500)   # 预测值
target = tf.constant(0.)    # 真实值


# 计算L1_loss
l1_y_val = tf.abs(target - x_val)   # 绝对值
l1_y_out = sess.run(l1_y_val)  # 用这个函数打开计算图

# 计算L2_loss
l2_y_val = tf.square(target - x_val)  # 平方
l2_y_out = sess.run(l2_y_val)  # 用这个函数打开计算图


# 打开计算图输出x_val，用来画图
# 用画图来体现损失函数的特点
x_array = sess.run(x_val)
plt.plot(x_array, l1_y_out, 'b', label='L1_loss')
plt.plot(x_array, l2_y_out, 'r', label='L2_loss')
# 用legend（）显示图例
plt.legend()

# plt.savefig("./image_test/mse.png", format="png")
plt.show()

