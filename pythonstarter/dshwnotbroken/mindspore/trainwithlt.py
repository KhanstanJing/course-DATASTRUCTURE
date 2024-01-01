from mynet import *
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
import mindspore.nn as nn
import os
import numpy as np
from mindspore import context
from mindspore import Model, dataset
from mindspore.train.callback import ReduceLROnPlateau
import matplotlib.pyplot as plt
import mindspore as ms
from mindspore.nn import SoftmaxCrossEntropyWithLogits

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")  # 或 "CPU"

# 文件的保存路径
save_dir = r'D:\pythonstarter\dshwnotbroken\mindspore\data'

# 加载数据和标签
train_data = np.load(os.path.join(save_dir, 'train_data.npy'))
test_data = np.load(os.path.join(save_dir, 'test_data.npy'))
val_data = np.load(os.path.join(save_dir, 'val_data.npy'))
train_label = np.load(os.path.join(save_dir, 'train_label.npy'))
test_label = np.load(os.path.join(save_dir, 'test_label.npy'))
val_label = np.load(os.path.join(save_dir, 'val_label.npy'))

train_size = len(train_data)
test_size = len(test_data)

#
train_dataset = dataset.NumpySlicesDataset({"data": train_data, "label": train_label}, shuffle=True)
train_dataset = train_dataset.batch(4, drop_remainder=True)
# train_loader = dataset.GeneratorDataset(train_dataset, shuffle=True)
print(type(train_dataset))
print(train_dataset.batch)
eval_dataset = dataset.NumpySlicesDataset({"data": val_data, "label": val_label}, shuffle=False)
eval_dataset = eval_dataset.batch(4, drop_remainder=True)
# eval_loader = dataset.GeneratorDataset(train_dataset, shuffle=True)

# 实例化CNN网络
net = MyNet()
loss_fn = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction= 'mean')
optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=0.001)
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=2, verbose=True, factor=0.3, min_lr=0.000001)

# 创建其他回调函数，比如LossMonitor和TimeMonitor
loss_monitor = LossMonitor()
time_monitor = TimeMonitor(data_size=train_dataset.get_dataset_size())

# 创建ModelCheckpoint回调函数，用于保存模型
config_ck = CheckpointConfig(save_checkpoint_steps=train_dataset.get_dataset_size(), keep_checkpoint_max=12)
ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", directory=r"D:\pythonstarter\dshwnotbroken\mindspore\checkpoint", config=config_ck)

# 使用Model接口
# model = Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics={"accuracy"})
# model._train(epoch=3, train_dataset=train_dataset, callbacks=[learning_rate_reduction, loss_monitor, time_monitor, ckpoint_cb])

# 使用TrainOneStepCell自定义网络
loss_net = nn.WithLossCell(net, loss_fn) # 包含损失函数的Cell
train_net = nn.TrainOneStepCell(loss_net, optimizer)

for i in range(train_size):
    for image, label in train_dataset:
        train_net.set_train()
        res = train_net(image, label) # 执行网络的单步训练
        
        loss = loss_fn(outputs)
        print(loss_fn)

loss_list = []
acc_list = []
epochs = 100

for epoch in range(epochs):
    print('epoch = ', epoch+1, '/', epochs)

    train_total_loss = 0.0
    test_total_loss = 0.0

    train_total_acc = 0.0
    test_total_acc = 0.0

    for index, data in enumerate(train_dataset):
        print('epoch = ', epoch+1, '/', epochs, 'step', index+1)
        inputs, labels = data

        # 清空梯度
        optimizer.zero_grad()

        outputs = MyNet(inputs)
        # print('the inputs is ', inputs, 'the labels is', labels)
        # print((labels.shape))
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        index = ms.argmax(outputs, axis=1)
        # print(outputs)
        # print(index)
        # print(_)
        acc = ms.sum(index == labels).asnumpy().item()

        train_total_loss += loss.item()
        train_total_acc += acc
    print('train loss is', train_total_loss, 'train accuracy is', train_total_acc / train_size)
    loss_list.append(train_total_loss)
    acc_list.append(train_total_acc)

# torch.save(myModel, 'model/model100.pth')

# 包含每轮损失的列表 loss_list 和准确率列表 acc_list
epochs = range(1, len(loss_list) + 1)

# 绘制损失图表
plt.plot(epochs, loss_list, label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制准确率图表
plt.plot(epochs, acc_list, label='Training Accuracy')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

eval_result = net.eval(eval_dataset)
print(eval_result)


