import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载波士顿房屋数据集
data_url = 'housing.csv'
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 数据准备
x = data
y = target

# 将数据集划分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 对特征进行标准化处理
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 线性回归模型
class LinearRegressionManual:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, x, y):
        # 添加截距项
        x_b = np.c_[np.ones((x.shape[0], 1)), x]
        # 计算正规方程
        theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
        self.intercept_ = theta_best[0]
        self.coef_ = theta_best[1:]

    def predict(self, x):
        x_b = np.c_[np.ones((x.shape[0], 1)), x]
        return x_b.dot(np.r_[self.intercept_, self.coef_])

# 创建线性回归模型实例
model = LinearRegressionManual()

# 模型训练
model.fit(x_train_scaled, y_train)

# 模型预测
y_pred = model.predict(x_test_scaled)

# 计算均方误差
mse = ((y_pred - y_test) ** 2).mean()
print("MSE：", mse)

# 结果可视化
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predict Price')
plt.title('Linear Regression')
plt.show()
