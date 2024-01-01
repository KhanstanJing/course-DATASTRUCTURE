import matplotlib.pyplot as plt
import numpy as np
import pickle

pacc300 = r'D:\pythonstarter\dshwwithpytorch\loss&accuracy\acc300.pkl'
pacc250 = r'D:\pythonstarter\dshwwithpytorch\loss&accuracy\acc250.pkl'
pacc200 = r'D:\pythonstarter\dshwwithpytorch\loss&accuracy\acc200.pkl'
pacc150 = r'D:\pythonstarter\dshwwithpytorch\loss&accuracy\acc150.pkl'
pacc100 = r'D:\pythonstarter\dshwwithpytorch\loss&accuracy\acc100.pkl'
pacc50 = r'D:\pythonstarter\dshwwithpytorch\loss&accuracy\acc50.pkl'

ploss300 = r'D:\pythonstarter\dshwwithpytorch\loss&accuracy\loss300.pkl'
ploss250 = r'D:\pythonstarter\dshwwithpytorch\loss&accuracy\loss250.pkl'
ploss200 = r'D:\pythonstarter\dshwwithpytorch\loss&accuracy\loss200.pkl'
ploss150 = r'D:\pythonstarter\dshwwithpytorch\loss&accuracy\loss150.pkl'
ploss100 = r'D:\pythonstarter\dshwwithpytorch\loss&accuracy\loss100.pkl'
ploss50 = r'D:\pythonstarter\dshwwithpytorch\loss&accuracy\loss50.pkl'

# 从Pickle文件中加载数据
with open(pacc50, 'rb') as file:
    acc50 = pickle.load(file)
with open(pacc100, 'rb') as file:
    acc100 = pickle.load(file)
with open(pacc150, 'rb') as file:
    acc150 = pickle.load(file)
with open(pacc200, 'rb') as file:
    acc200 = pickle.load(file)
with open(pacc250, 'rb') as file:
    acc250 = pickle.load(file)
with open(pacc300, 'rb') as file:
    acc300 = pickle.load(file)

with open(ploss50, 'rb') as file:
    loss50 = pickle.load(file)
with open(ploss100, 'rb') as file:
    loss100 = pickle.load(file)
with open(ploss150, 'rb') as file:
    loss150 = pickle.load(file)
with open(ploss200, 'rb') as file:
    loss200 = pickle.load(file)
with open(ploss250, 'rb') as file:
    loss250 = pickle.load(file)
with open(ploss300, 'rb') as file:
    loss300 = pickle.load(file)
# x = np.arange(1, 51)  # 十二个样本的横坐标
# plt.plot(x, acc50, color= '#1f77b4', label='50epochs')
x = np.arange(1, 101)  # 十二个样本的横坐标
plt.plot(x, acc100, color= '#ff7f0e', label='100epochs')
# x = np.arange(1, 151)  # 十二个样本的横坐标
# plt.plot(x, acc150, color= '#9467bd', label='150epochs')
x = np.arange(1, 201)  # 十二个样本的横坐标
plt.plot(x, acc200, color= '#bcbd22', label='200epochs')
# x = np.arange(1, 251)  # 十二个样本的横坐标
# plt.plot(x, acc250, color= '#17becf', label='250epochs')
x = np.arange(1, 301)  # 十二个样本的横坐标
plt.plot(x, acc300, color= '#2ca02c', label='300epochs')
# 添加标题和标签
plt.xlabel('epochs')
plt.ylabel('accuracy')
# 添加图例
plt.legend()
# 保存图像
plt.savefig('./plots/accuracy3.png')
# 显示图形
plt.show()

# 绘制折线图
# plt.plot(50, loss50, color= '#1f77b4', label='50epochs')
x = np.arange(1, 101)  # 十二个样本的横坐标
plt.plot(x, loss100, color= '#ff7f0e', label='100epochs')
# plt.plot(150, loss150, color= '#9467bd', label='150epochs')
x = np.arange(1, 201)  # 十二个样本的横坐标
plt.plot(x, loss200, color= '#bcbd22', label='200epochs')
# plt.plot(250, loss250, color= '#17becf', label='250epochs')
x = np.arange(1, 301)  # 十二个样本的横坐标
plt.plot(x, loss300, color= '#2ca02c', label='300epochs')
# 添加标题和标签
plt.xlabel('epochs')
plt.ylabel('loss')
# 添加图例
plt.legend()
# 保存图像
plt.savefig('./plots/loss3.png')
# 显示图形
plt.show()