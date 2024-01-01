from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mynet import *
import mindspore.nn as nn
import numpy as np
import os
import mindspore

# 定义神经网络
net = MyNet()
criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

# 加载checkpoint文件
# checkpoint_path = r'D:\pythonstarter\dshwnotbroken\mindspore\checkpoint\checkpoint_lenet_2-6_1304.ckpt'
checkpoint_path = r'D:\pythonstarter\dshwnotbroken\mindspore\checkpoint\checkpoint_lenet_4-1_5216.ckpt'

param_dict = load_checkpoint(checkpoint_path)

# 将参数加载到神经网络中
load_param_into_net(net, param_dict)

# 文件的保存路径
save_dir = r'D:\pythonstarter\dshwnotbroken\mindspore\data'

# 加载checkpoint文件中的参数
test_data = np.load(os.path.join(save_dir, 'test_data.npy'))
test_label = np.load(os.path.join(save_dir, 'test_label.npy'))

# 使用mindspore.Tensor将NumPy数组转换为MindSpore张量
test_data = mindspore.Tensor(test_data, dtype=mindspore.float32)
test_label = mindspore.Tensor(test_label, dtype=mindspore.int32)

# 将网络设置为评估模式
# net.set_train(False)

output = net(test_data)
'''
print(type(output))
print(len(output))
for i in range(624):
    bacteria = virus = normal = 0
    if output[i, 0] >= output[i, 1] & output[i, 0] >= output[i, 2]:
        bacteria = bacteria + 1
    elif output[i, 0] >= output[i, 1] & output[i, 0] <= output[i, 2]:
        normal = normal + 1
    else:
        virus = virus + 1
print(bacteria, virus, normal)
'''
#print(output)
loss = criterion(output, test_label)

# 进行评估（例如，准确度）
accuracy = np.mean((np.argmax(output.asnumpy(), axis=1) == test_label.asnumpy()).astype(np.float32))

# 打印结果
print("Loss of the model is - ", loss.asnumpy())
print("Accuracy of the model is - ", accuracy * 100, "%")