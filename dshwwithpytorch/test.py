from torch.optim import SGD
import numpy as  np
import os
from net import *
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torch.optim import Adam
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

# 指定文件夹
test_data = np.load(os.path.join(save_dir, 'test_data.npy'))
test_size = len(test_data)
test_label = np.load(os.path.join(save_dir, 'test_label.npy'))
test_dataset = torch.from_numpy(test_data)
test_labels = torch.from_numpy(test_label)
test_labels = test_labels.to(torch.long)
testdataset = TensorDataset(test_dataset, test_labels)
testloader = DataLoader(testdataset, batch_size=1, shuffle=True)

classes = ['bacteria', 'virus', 'normal']
myModel = MyModel()
use_gpu = torch.cuda.is_available()
if(use_gpu):
    print("GPU可用")
    myModel = myModel.cuda()

optimizer = SGD(myModel.parameters(), lr = 0.01)
# optimizer = Adam(myModel.parameters(), lr=0.01)
lossf = torch.nn.CrossEntropyLoss()
epochs = 175
acc_list = []
test_total_acc = 0.0

for epoch in range(epochs):
    print('epoch = ', epoch+1, '/', epochs)
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

    with torch.no_grad():
        test_total_acc = 0
        for index, data in enumerate(testloader):
            inputs, labels = data
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = myModel(inputs)
            value, ind = torch.max(outputs, 1)
            #for i in ind:
            pre_val = classes[ind]
            print('预测概率：{}， 预测下标：{}， 预测结果：{}'.format(value.item(), ind, pre_val))

            _, index = torch.max(outputs, 1)
            acc = torch.sum(index == labels).item()
            test_total_acc += acc
        accrate = test_total_acc / test_size
        print('test accuracy is', accrate, test_total_acc, test_size)
        acc_list.append(accrate)
    # 指定文件夹路径
    folder_path = r'D:\pythonstarter\dshwwithpytorch\loss&accuracy'
    # 将列表保存到文件
    accperepoch = os.path.join(folder_path, 'accperepoch.pkl')
    with open(accperepoch, 'wb') as file:
        pickle.dump(acc_list, file)
    print(accperepoch)
x = np.arange(1, 176)
# 绘制准确率图表
plt.plot(x, acc_list, label='Accuracy per epoch')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(r'D:\pythonstarter\dshwwithpytorch\plots\accerepochwithadam.png')
plt.show()
