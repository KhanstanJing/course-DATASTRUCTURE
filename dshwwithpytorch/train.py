from torch.optim import SGD
import numpy as  np
import os
from net import *
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pickle

save_dir = r"D:\pythonstarter\dshwwithpytorch\mydataset"

# 加载数据和标签
train_data = np.load(os.path.join(save_dir, 'train_data.npy'))
train_size = len(train_data)

train_label = np.load(os.path.join(save_dir, 'train_label.npy'))

# 将 numpy 数组转换为 PyTorch 张量
train_dataset = torch.from_numpy(train_data)
train_labels = torch.from_numpy(train_label)
train_labels = train_labels.to(torch.long)

# 创建数据集和数据加载器
traindataset = TensorDataset(train_dataset, train_labels)
trainloader = DataLoader(traindataset, batch_size=4, shuffle=True)

myModel = MyModel()
use_gpu = torch.cuda.is_available()
if(use_gpu):
    print("GPU可用")
    myModel = myModel.cuda()

optimizer = SGD(myModel.parameters(), lr = 0.01)
lossf = torch.nn.CrossEntropyLoss()

loss_list = []
acc_list = []

epochs = 50

for epoch in range(epochs):
    print('epoch = ', epoch+1, '/', epochs)

    train_total_loss = 0.0
    test_total_loss = 0.0

    train_total_acc = 0.0
    test_total_acc = 0.0

    for index, data in enumerate(trainloader):
        print('epoch = ', epoch+1, '/', epochs, 'step', index+1)
        inputs, labels = data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        outputs = myModel(inputs)

        loss = lossf(outputs, labels)
        loss.backward()
        optimizer.step()

        _, index = torch.max(outputs, 1)

        acc = torch.sum(index == labels).item()

        train_total_loss += loss.item()
        train_total_acc += acc
    accrate = train_total_acc / train_size
    print('train loss is', train_total_loss, 'train accuracy is', accrate)

    loss_list.append(train_total_loss)
    acc_list.append(accrate)
    # 指定文件夹路径
    folder_path = r'D:\pythonstarter\dshwwithpytorch\loss&accuracy'
    # 将列表保存到文件
    loss50 = os.path.join(folder_path, 'loss50.pkl')
    with open(loss50, 'wb') as file:
        pickle.dump(loss_list, file)
    acc50 = os.path.join(folder_path, 'acc50.pkl')
    with open(acc50, 'wb') as file:
        pickle.dump(acc_list, file)

torch.save(myModel, 'model/model50.pth')

# 包含每轮损失的列表 loss_list 和准确率列表 acc_list
epochs = range(1, len(loss_list) + 1)

# 绘制损失图表
plt.plot(epochs, loss_list, label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(r'D:\pythonstarter\dshwwithpytorch\plots\loss_plot50.png')  # 将图表保存为 loss_plot.png
plt.show()

# 绘制准确率图表
plt.plot(epochs, acc_list, label='Training Accuracy')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(r'D:\pythonstarter\dshwwithpytorch\plots\accuracy_plot50.png')  # 将图表保存为 loss_plot.png
plt.show()





