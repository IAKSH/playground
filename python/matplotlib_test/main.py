import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 尝试'fivethirtyeight'主题
plt.style.use("fivethirtyeight")

# 准备数据
x = np.linspace(0, 30, 300)
y = np.sin(x)

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

matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
m=50
x=np.random.rand(m)
y=np.random.rand(m)
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.scatter(x, y)
plt.title("默认情况下")

plt.show()

# s和原始数组大小一样
area=(30*np.random.rand(m))**2
plt.subplot(121)
plt.scatter(x, y,s=100)
plt.title("点大小相同")
plt.subplot(122)
plt.scatter(x, y,s=area)
plt.title("点大小不同")

plt.show()

# c和原始数组大小一样
colors=np.random.rand(m)
plt.subplot(121)
plt.scatter(x, y,s=area,c=colors)
plt.title("大小s和颜色c")
plt.subplot(122)
plt.title("颜色透明度alpha")
plt.scatter(x, y,s=area,c=colors,alpha=0.5)

plt.show()

# 点样式和线宽度
lines=np.zeros(m)+5
plt.subplot(121)
plt.title("点样式marker")
plt.scatter(x, y,s=area,c=colors,alpha=0.5,marker='+')
plt.subplot(122)
plt.title("点样式为闭合的情况下，线宽度")
plt.scatter(x, y,s=area,c=colors,alpha=0.5,marker='o',linewidths=lines)

plt.show()