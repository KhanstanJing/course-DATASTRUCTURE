import matplotlib.pyplot as plt
import numpy as np
import pickle

# 指定Pickle文件路径
pllmeory1 = r'D:\pythonstarter\dshwwithpytorch\time&memory\llmemory1.pkl'
pllmeory2 = r'D:\pythonstarter\dshwwithpytorch\time&memory\llmemory2.pkl'
pllmeory3 = r'D:\pythonstarter\dshwwithpytorch\time&memory\llmemory3.pkl'
pltmeory1 = r'D:\pythonstarter\dshwwithpytorch\time&memory\ltmemory1.pkl'
pltmeory2 = r'D:\pythonstarter\dshwwithpytorch\time&memory\ltmemory2.pkl'
pltmeory3 = r'D:\pythonstarter\dshwwithpytorch\time&memory\ltmemory3.pkl'
plltime1 = r'D:\pythonstarter\dshwwithpytorch\time&memory\lltime1.pkl'
plltime2 = r'D:\pythonstarter\dshwwithpytorch\time&memory\lltime2.pkl'
plltime3 = r'D:\pythonstarter\dshwwithpytorch\time&memory\lltime3.pkl'
plttime1 = r'D:\pythonstarter\dshwwithpytorch\time&memory\lttime1.pkl'
plttime2 = r'D:\pythonstarter\dshwwithpytorch\time&memory\lttime2.pkl'
plttime3 = r'D:\pythonstarter\dshwwithpytorch\time&memory\lttime3.pkl'

# 从Pickle文件中加载数据
with open(pllmeory1, 'rb') as file:
    llmeory1 = pickle.load(file)
with open(pllmeory2, 'rb') as file:
    llmeory2 = pickle.load(file)
with open(pllmeory3, 'rb') as file:
    llmeory3 = pickle.load(file)
with open(pltmeory1, 'rb') as file:
    ltmeory1 = pickle.load(file)
with open(pltmeory2, 'rb') as file:
    ltmeory2 = pickle.load(file)
with open(pltmeory3, 'rb') as file:
    ltmeory3 = pickle.load(file)
with open(plltime1, 'rb') as file:
    lltime1 = pickle.load(file)
with open(plltime2, 'rb') as file:
    lltime2 = pickle.load(file)
with open(plltime3, 'rb') as file:
    lltime3 = pickle.load(file)
with open(plttime1, 'rb') as file:
    lttime1 = pickle.load(file)
with open(plttime2, 'rb') as file:
    lttime2 = pickle.load(file)
with open(plttime3, 'rb') as file:
    lttime3 = pickle.load(file)

# 生成示例数据
x = np.arange(1, 13)  # 十二个样本的横坐标
# #1f77b4是蓝线，代表线性表
# 绘制折线图
plt.plot(x, llmeory1, color= '#1f77b4', label='linked list', marker='o')
plt.axhline(y=np.mean(llmeory1), color= '#1f77b4', linestyle='--', label='mean of linked list')
plt.text(6, 0.29, f'{round(np.mean(llmeory1), 4)}', color='#1f77b4', ha='center', va='bottom')
plt.plot(x, ltmeory1, color= '#ff7f0e', label='linear table', marker='s')
plt.axhline(y=np.mean(ltmeory1), color= '#ff7f0e', linestyle='--', label='mean of linear table')
plt.text(6, 0.095, f'{round(np.mean(ltmeory1), 4)}', color='#ff7f0e', ha='center', va='bottom')
# 添加标题和标签
plt.xlabel('test')
plt.ylabel('MB')
# 添加图例
plt.legend()
# 保存图像
plt.savefig('./plots/memory1.png')
# 显示图形
plt.show()

# 绘制折线图
plt.plot(x, llmeory2, color= '#1f77b4', label='linked list', marker='o')
plt.axhline(y=np.mean(llmeory2), color= '#1f77b4', linestyle='--', label='linked list')
plt.text(6.3, 115.449, f'{round(np.mean(llmeory2), 4)}', color='#1f77b4', ha='center', va='bottom')
plt.plot(x, ltmeory2, color= '#ff7f0e', label='linear table', marker='s')
plt.axhline(y=np.mean(ltmeory2), color= '#ff7f0e', linestyle='--', label='mean of linear table')
plt.text(5.5, 114.7, f'{round(np.mean(ltmeory2), 4)}', color='#ff7f0e', ha='center', va='bottom')
# 添加标题和标签
#plt.title('将图片信息加载到数据结构中后内存占用增加', fontproperties=font_prop)
plt.xlabel('test')
plt.ylabel('MB')
# 添加图例
plt.legend()
# 保存图像
plt.savefig('./plots/memory2.png')
# 显示图形
plt.show()

# 绘制折线图
plt.plot(x, llmeory3, color= '#1f77b4', label='linked list', marker='o')
plt.axhline(y=np.mean(llmeory3), color= '#1f77b4', linestyle='--', label='mean of linked list')
plt.text(12, 0.128, f'{round(np.mean(llmeory3), 4)}', color='#1f77b4', ha='center', va='bottom')
plt.plot(x, ltmeory3, color= '#ff7f0e', label='linear table', marker='s')
plt.axhline(y=np.mean(ltmeory3), color= '#ff7f0e', linestyle='--', label='mean of linear table')
plt.text(12, 0.0280, f'{round(np.mean(ltmeory3), 4)}', color='#ff7f0e', ha='center', va='bottom')
# 添加标题和标签
#plt.title('将图片信息加载到数据结构中后内存占用增加', fontproperties=font_prop)
plt.xlabel('test')
plt.ylabel('MB')
# 添加图例
plt.legend()
# 保存图像
plt.savefig('./plots/memory3.png')
# 显示图形
plt.show()


# 绘制折线图
plt.plot(x, lltime1, color= '#1f77b4', label='linked list', marker='o')
plt.axhline(y=np.mean(lltime1), color= '#1f77b4', linestyle='--', label='mean of linked list')
plt.text(11, 0.002085, f'{round(np.mean(lltime1), 6)}', color='#1f77b4', ha='center', va='top')
plt.plot(x, lttime1, color= '#ff7f0e', label='linear table', marker='s')
plt.axhline(y=np.mean(lttime1), color= '#ff7f0e', linestyle='--', label='mean of linear table')
plt.text(11, .00063, f'{round(np.mean(lttime1), 6)}', color='#ff7f0e', ha='center', va='top')
# 添加标题和标签
#plt.title('将图片信息加载到数据结构中后内存占用增加', fontproperties=font_prop)
plt.xlabel('test')
plt.ylabel('s')
# 添加图例
plt.legend()
# 保存图像
plt.savefig('./plots/time1.png')
# 显示图形
plt.show()

# 绘制折线图
plt.plot(x, lltime2, color= '#1f77b4', label='linked list', marker='o')
plt.axhline(y=np.mean(lltime2), color= '#1f77b4', linestyle='--', label='mean of linked list')
plt.text(7.3, 20.6, f'{round(np.mean(lltime2), 4)}', color='#1f77b4', ha='center', va='top')
plt.plot(x, lttime2, color= '#ff7f0e', label='linear table', marker='s')
plt.axhline(y=np.mean(lttime2), color= '#ff7f0e', linestyle='--', label='mean of linear table')
plt.text(7.3, 23.3, f'{round(np.mean(lttime2), 4)}', color='#ff7f0e', ha='center', va='top')
# 添加标题和标签
#plt.title('将图片信息加载到数据结构中后内存占用增加', fontproperties=font_prop)
plt.xlabel('test')
plt.ylabel('s')
# 添加图例
plt.legend()
# 保存图像
plt.savefig('./plots/time2.png')
# 显示图形
plt.show()

# 绘制折线图
plt.plot(x, lltime3, color= '#1f77b4', label='linked list', marker='o')
plt.axhline(y=np.mean(lltime3), color= '#1f77b4', linestyle='--', label='mean of linked list')
plt.text(8.48, 0.000645, f'{round(np.mean(lltime3), 6)}', color='#1f77b4', ha='center', va='top')
plt.plot(x, lttime3, color= '#ff7f0e', label='linear table', marker='s')
plt.axhline(y=np.mean(lttime3), color= '#ff7f0e', linestyle='--', label='mean of linear table')
plt.text(5.55, 0.00025, f'{round(np.mean(lttime3), 6)}', color='#ff7f0e', ha='center', va='top')
# 添加标题和标签
#plt.title('将图片信息加载到数据结构中后内存占用增加', fontproperties=font_prop)
plt.xlabel('test')
plt.ylabel('s')
# 添加图例
plt.legend()
# 保存图像
plt.savefig('./plots/time3.png')
# 显示图形
plt.show()

