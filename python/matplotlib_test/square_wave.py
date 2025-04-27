import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def lowmist_func(x,delta):
    return 1 - 2 * math.ceil((x / 500) - math.floor(x / 500) - delta)

def original_func(x,delta):
    if x % 1000 < 1000 * delta:
        return 1
    else:
        return 0

'''
for (int i = 0; i < samplesPerCycle; i++) {
            if ((i + phaseSamples) % samplesPerCycle < halfSamples) {
                data[i] = (short) (Short.MAX_VALUE * amplitude);
            } else {
                data[i] = (short) (Short.MIN_VALUE * amplitude);
            }
        }
'''

# 尝试'fivethirtyeight'主题
plt.style.use("fivethirtyeight")

# 准备数据

x = []
y = []

for i in range(1000):
    x.append(i)
    y.append(lowmist_func(i,0.2))

print("ploting")

# 调用ax.plot()创建散点图，跟创建曲线图一样
# 第3个(可选)参数'fmt'控制几何图形，'-'代表实线，'o'代表点(实心圆)
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(
    x, y, "-o",  # 同时创建曲线和散点
    markersize=1,  # 控制点的大小
    linewidth=1,  # 控制线的大小
    markerfacecolor="red",  # 控制圆圈填充的颜色
    markeredgecolor="red",  # 控制圆圈边缘的颜色
    markeredgewidth=4  # 控制圆圈边缘的大小
)

plt.show()
