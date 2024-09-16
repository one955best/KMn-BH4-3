import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import time
import pattens as pt

pattens_data = pt.pattens_data_ABX3
pattens_predict = pt.pattens_predict_ABX3

path = 'D:/FJW/project_artical/new_ABX3.txt'

data = pd.read_csv(path, names=pattens_data)

cols = data.shape[1]
X = data.iloc[:, 3: cols - 1]
y = data.iloc[:, cols - 1: cols]
X = np.array(X)
y = np.array(y).ravel()

params = {'n_estimators': 300,  # 弱分类器的个数
          'max_depth': 3,  # 弱分类器（CART回归树）的最大深度
          'min_samples_split': 5,  # 分裂内部节点所需的最小样本数
          'learning_rate': 0.05,  # 学习率
          'loss': 'squared_error'}  # 损失函数类型

t0 = time.time()
A, M, N = [], [], []

for k in range(20, 350):
    m, n = 0, 0
    i = 0
    params = {'n_estimators': k,  # 弱分类器的个数
                  'max_depth': 3,  # 弱分类器（CART回归树）的最大深度
                  'min_samples_split': 5,  # 分裂内部节点所需的最小样本数
                  'learning_rate': 0.05,  # 学习率
                  'loss': 'squared_error'}  # 损失函数类型
    while i < 50:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        clf = GradientBoostingRegressor(**params)
        clf.fit(X_train, y_train)
        m = m + clf.score(X_test, y_test)
        n = n + clf.score(X_train, y_train)
        i = i + 1
    A.append(k)
    M.append(m / 50)
    N.append(n / 50)
    print(k)

time_fit = time.time() - t0

df = pd.DataFrame({'n_estimators': A, 'M': M, 'N': N})
df.to_csv('D:/FJW/project_artical/R^2_ABX3_GDBT.csv', index=False, sep=',')
print(time_fit)


plt.plot(A, M, label='M')
plt.plot(A, N, label='N')
plt.xlabel('k')
plt.ylabel('R^2')
plt.show()