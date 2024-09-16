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

pattens_data = pt.pattens_data_A2BCX6
pattens_predict = pt.pattens_predict_A2BCX6

path = 'new_A2BCX6.txt'

data = pd.read_csv(path, names=pattens_data)

cols = data.shape[1]
X = data.iloc[:, 4: cols - 1]
y = data.iloc[:, cols - 1: cols]
X = np.array(X)
y = np.array(y).ravel()

params = {'n_estimators': 206,  # 弱分类器的个数
          'max_depth': 3,  # 弱分类器（CART回归树）的最大深度
          'min_samples_split': 5,  # 分裂内部节点所需的最小样本数
          'learning_rate': 0.05,  # 学习率
          'loss': 'squared_error'}  # 损失函数类型

y_test_all = []
y_pred_all = []
test_score = 0

i, m, n = 0, 0, 0
while i < 50:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = GradientBoostingRegressor(**params)
    clf.fit(X_train, y_train)
    m = m + clf.score(X_test, y_test)
    n = n + clf.score(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 保存预测值和真实值
    y_test_all.extend(y_test)
    y_pred_all.extend(y_pred)
    i = i + 1
df = pd.DataFrame({'y_test_all': y_test_all, 'y_pred_all': y_pred_all})
df.to_csv('C:/A2BCX6_train_GDBT_best.csv', index=False, sep=',')

test_score = m / 50
print(test_score)


plt.figure(figsize=(10, 6))
plt.scatter(y_test_all, y_pred_all, color="blue", alpha=0.6)
plt.plot([min(y_test_all), max(y_test_all)], [min(y_test_all), max(y_test_all)], 'r--', lw=2)  # 画对角线，用于表示理想预测
plt.title("True vs Predicted Values on Test Set")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.grid(True)
plt.show()
