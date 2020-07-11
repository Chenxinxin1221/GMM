import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings("ignore")

plt.style.use('seaborn')

# # 预处理
# # data = np.fromfile(url, dtype=np.int32)
# # data = np.loadtxt(url, dtype=float)
# data_old = pd.read_csv(url, encoding='gbk')
# # col = data_old.select_dtypes(exclude='object').columns
# # name = [a for a in col if a not in ['clinic number']]
# # data = data_old[name]
# df = pd.DataFrame(data_old)
# # # 查看哪些数值有缺失值
# print(df.isnull().sum() > 0)
# # 对所有数据的缺失值用均值填充
# for column in list(df.columns[df.isnull().sum() > 0]):
#     mean_val = df[column].mean()
#     df[column].fillna(mean_val, inplace=True)
# # 查看是否还有缺失值
# # print(df.isnull().sum() > 0)
# df.to_csv('D:/Heart/datanonull.csv')
# # print(data)
# return df

# 生成数据
def load_X(url):
    data_old = pd.read_csv(url)
    df = pd.DataFrame(data_old)
    # 判断是否有缺失值
    # print(df.isnull().sum() > 0)
    # data = np.loadtxt(url, dtype=float)
    # data = pd.read_csv(url, encoding='gbk')
    # print(df)
    data = 10*np.array(df)
    # print(data)
    data = data.astype(np.int64)
    # print(data)
    data = data.astype(np.float64)
    # print(data)


    for i in range(data.shape[1]):   # 对数据每一列进行归一化
        sums = data[:, i]
        data[:, i] = (data[:, i] - np.min(data[:, i])) / (np.max(data[:, i]) - np.min(data[:, i]))

    return data


# 更新W
def update_W(X, Mu, Var, Pi):
    n_points, n_clusters = len(X), len(Pi)
    # print(n_points)    # 684
    # print(n_clusters)  # 2
    pdfs = np.zeros(((n_points, n_clusters)))   # 684*2
    # print(pdfs)
    # print(X)
    # print(Pi)   # 0.5,0.5
    for i in range(n_clusters):
        # +0.01 * np.identity(X.shape(1)))
        print(multivariate_normal.pdf(X, Mu[i], np.diag(Var[i])))
        pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Mu[i], np.diag(Var[i]))
    W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
    # print(W)
    return W


# 更新pi
def update_Pi(W):
    Pi = W.sum(axis=0) / W.sum()
    return Pi


# 计算log似然函数
def logLH(X, Pi, Mu, Var):
    n_points, n_clusters = len(X), len(Pi)
    pdfs = np.zeros(((n_points, n_clusters)))
    for i in range(n_clusters):
        pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Mu[i], np.diag(Var[i]))
    return np.mean(np.log(pdfs.sum(axis=1)))


# # 画出聚类图像
# def plot_clusters(X, Mu, Var, Mu_true=None, Var_true=None):
#     colors = ['b', 'r']
#     n_clusters = len(Mu)
#     plt.figure(figsize=(10, 8))
#     plt.axis([-10, 15, -5, 15])
#     plt.scatter(X[:, 0], X[:, 1], s=5)
#     ax = plt.gca()
#     for i in range(n_clusters):
#         plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'ls': ':'}
#         ellipse = Ellipse(Mu[i], 3 * Var[i][0], 3 * Var[i][1], **plot_args)
#         ax.add_patch(ellipse)
#     if (Mu_true is not None) & (Var_true is not None):
#         for i in range(n_clusters):
#             plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'alpha': 0.5}
#             ellipse = Ellipse(Mu_true[i], 3 * Var_true[i][0], 3 * Var_true[i][1], **plot_args)
#             ax.add_patch(ellipse)
#     plt.show()


# 更新Mu
def update_Mu(X, W):
    n_clusters = W.shape[1]
    Mu = np.zeros((n_clusters, 14))     # 均值的维度为2*14
    # print(Mu)
    # print(X)
    # print(W)
    for i in range(n_clusters):
        Mu[i] = np.average(X, axis=0, weights=W[:, i])
    return Mu


# 更新Var
def update_Var(X, Mu, W):
    n_clusters = W.shape[1]
    Var = np.zeros((n_clusters, 14))
    for i in range(n_clusters):
        Var[i] = np.average((X - Mu[i]) ** 2, axis=0, weights=W[:, i])
    return Var


if __name__ == '__main__':
    # 生成数据
    X = load_X('D:/Heart/datanonull.csv')
    # print(X)
    # 初始化
    n_clusters = 2
    n_points = len(X)   # 返回的数据集X的长度 684
    # print(n_points)
    Mu = [[0, 2, 1, 0, 0, 1, 2, 1, 2, 1, 1, 2, 5, 1], [2, 2, 3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]]
    Var = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    W = np.ones((n_points, n_clusters)) / n_clusters
    # print(W)
    Pi = W.sum(axis=0) / W.sum()
    # # 迭代
    loglh = []
    for i in range(5):
    #     # plot_clusters(X, Mu, Var)
    #     loglh.append(logLH(X, Pi, Mu, Var))
        W = update_W(X, Mu, Var, Pi)
        Pi = update_Pi(W)
        Mu = update_Mu(X, W)
        # print('log-likehood:%.3f' % loglh[-1])
        Var = update_Var(X, Mu, W)
    # print(W)

