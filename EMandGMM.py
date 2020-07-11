# GMM:软聚类模型 k-means的推广
# 1、需要估计每个高斯分布的均值和方差，从最大似然的角度就是：给定数据集，给定GMM中有k簇，找到k组均值和方差最大化似然函数
# 2、直接计算很困难，于是引入隐变量（每个样本属于每一簇的概率）
# 3、EM算法。E步骤：更新隐变量W
#            M步骤：更新GMM中各高斯分布的参量（均值、方差）


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal  # 多元正态分布（就是多维数据的正态分布）

plt.style.use('seaborn')

# 生成2000个二维模拟数据 （数据是二维的就代表数据是有两个特征）
# 400个来自N1，600个来自N2，1000个来自N3
# 第一簇的数据
num1, mean1, var1 = 400, [0.5, 0.5], [1, 3]
X1 = np.random.multivariate_normal(mean1, np.diag(var1), num1)    # np.diag(a):a为一维数组，则输出以它们为对角线的矩阵;a为二维数组，则输出矩阵的对角线元素
# print(X1)
# 第二簇的数据
num2, mean2, var2 = 600, [5.5, 2.5], [2, 2]
X2 = np.random.multivariate_normal(mean2, np.diag(var2), num2)
# 第三簇的数据
num3, mean3, var3 = 1000, [1, 7], [6, 2]
X3 = np.random.multivariate_normal(mean3, np.diag(var3), num3)
# 合并在一起
X = np.vstack((X1, X2, X3))

# print(X1)  # 得到的是以均值0.5正态分布、和以均值0.5正态分布的值
# print(len(X1))
# print(X)
# print(len(X))

# 显示数据
plt.figure(figsize=(10, 8))
plt.axis([-10, 15, -5, 15])
plt.scatter(X1[:, 0], X1[:, 1], c='b', s=5)  # 第一列和第二列的数据
plt.scatter(X2[:, 0], X2[:, 1], c='g', s=5)
plt.scatter(X3[:, 0], X3[:, 1], c='r', s=5)
plt.show()


# 先初始化模型参数（先假设的，需去拟合的，不是真实的）
n_clusters = 3  # 簇的个数
n_points = len(X)  # 样本的个数
Mean = [[0, 0], [6, 0], [0, 9]]  # 初始化每个高斯分布的均值
Var = [[1, 1], [1, 2], [1, 1]]  # 初始化每个高斯分布的方差
W = np.ones((n_points, n_clusters)) / n_clusters  # 每个样本属于每一簇的概率（维度为n_points * n_clusters）
# print(W)
Pi = W.sum(axis=0) / W.sum()  # 每一簇的占比（GMM中每个高斯密度的权重）
# print(Pi)

# E步骤
# E步骤目的：更新隐变量W（每个样本属于每一簇的概率）
def update_W(X, Mean, Var, Pi):
    n_points, n_clusters = len(X), len(Pi)    # 样本数、簇数
    pdfs = np.zeros((n_points, n_clusters))   # 初始化隐变量
    # X是两列样本，对这个样本进行pdf，分为三簇
    for i in range(n_clusters):
        # print(multivariate_normal.pdf(X, Mean[i], np.diag(Var[i])))

        # multivariate_normal.pdf 得到X在Mean[i]取点值附近的可能性(即是输入X到此高斯分布得到的pdf概率密度函数)
        pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Mean[i], np.diag(Var[i]))   # 多维特征输入X，但得到的是一列的pdf
    W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)    # 除以每行的总值(先转换为1列)
    return W     # W的维度是（n_points * n_clusters）


def update_Pi(W):
    Pi = W.sum(axis=0) / W.sum()
    return Pi


# 对数似然函数
def logLH(X, Pi, Mean, Var):
    n_points, n_clusters = len(X), len(Pi)
    pdfs = np.zeros((n_points, n_clusters))
    for i in range(n_clusters):
        pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Mean[i], np.diag(Var[i]))    # 得到的是隐变量w
    # 得到的是所有样本属于每一簇的概率和的对数的均值（越大越好）
    return np.mean(np.log(pdfs.sum(axis=1)))    # np.log(a):以e为底，a的对数


# 可视化数据
def plot_clusters(X, Mean, Var, Mean_true=None, Var_True=None):   # Mean_true、Var_True代表的就是高斯分布的真值，就是要拟合这个参数
    colors = ['b', 'g', 'r']
    n_clusters = len(Mean)        # 均值有几组，就有几簇，高斯分布就有几个
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X[:, 0], X[:, 1], s=5)   # 这是数据的分布，我们就是要拟合这个分布
    ax = plt.gca()    # ax是当前的子图对象（就是数据）
    for i in range(n_clusters):
        # facecolor,线条宽度2，线条颜色，线条样式
        plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'ls': ':'}   # 画出需拟合的高斯分布（虚线）
        ellipse = Ellipse(Mean[i], 3 * Var[i][0], 3 * Var[i][1], **plot_args)   # Ellipse是椭圆，mean是椭圆的中心坐标；水平轴总长；垂直轴总长
        ax.add_patch(ellipse)
    if (Mean_true is not None) & (Var_True is not None):     # 画出真实的高斯分布（实线）
        for i in range(n_clusters):
            plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'alpha': 0.5}
            ellipse = Ellipse(Mean_true[i], 3 * Var_True[i][0], 3 * Var_True[i][1], **plot_args)
            ax.add_patch(ellipse)
    plt.show()


# M步骤
def update_Mean(X, W):
    n_clusters = W.shape[1]
    Mean = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        # 对每列数据带权求均值
        Mean[i] = np.average(X, axis=0, weights=W[:, i])  # X是n*2，每个Mean[i]维度是1*2
    return Mean


def update_Var(X, Mean, W):
    n_clusters = W.shape[1]
    Var = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        Var[i] = np.average((X - Mean[i]) ** 2, axis=0, weights=W[:, i])
    return Var


# 迭代求解
# true：实线， 需拟合：虚线
loglh = []
for i in range(10):
    plot_clusters(X, Mean, Var, [mean1, mean2, mean3], [var1, var2, var3])
    loglh.append(logLH(X, Pi, Mean, Var))
    # 更新迭代
    W = update_W(X, Mean, Var, Pi)
    Pi = update_Pi(W)
    Mean = update_Mean(X, W)
    print('log-likehood:%.3f' % loglh[-1])    # 似然函数必须单增才代表有效果
    Var = update_Var(X, Mean, W)
    print(Mean, Var)