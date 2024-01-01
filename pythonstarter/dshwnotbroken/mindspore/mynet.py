import mindspore.nn as nn
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P

class MyNet(nn.Cell):
    def __init__(self):
        super(MyNet, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, pad_mode='same', has_bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=0.9)
        self.relu1 = P.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')
        # print('1st layer succeed')

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, pad_mode='same', has_bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.9)
        self.relu2 = P.ReLU()
        self.dropout1 = nn.Dropout(p=0.8)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')
        # print('2nd layer succeed')

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, pad_mode='same', has_bias=False)
        self.bn3 = nn.BatchNorm2d(64, momentum=0.9)
        self.relu3 = P.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')
        # print('3rd layer succeed')

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, pad_mode='same', has_bias=False)
        self.bn4 = nn.BatchNorm2d(128, momentum=0.9)
        self.relu4 = P.ReLU()
        self.dropout2 = nn.Dropout(p=0.8)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')
        # print('4th layer succeed')

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, pad_mode='same', has_bias=False)
        self.bn5 = nn.BatchNorm2d(256, momentum=0.9)
        self.relu5 = P.ReLU()
        self.dropout3 = nn.Dropout(p=0.8)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')
        # print('5th layer succeed')

        # Define fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(6400, 128, weight_init=initializer('normal', (128, 6400)))
        self.relu6 = P.ReLU()
        self.dropout4 = nn.Dropout(p=0.2)
        # print('6th layer succeed')

        self.fc2 = nn.Dense(128, 3, weight_init=initializer('normal', (3, 128)))
        self.softmax = nn.Softmax(axis = 1)
        # self.sigmoid = P.Sigmoid()
        # print('last layer succeed')

        self.print_op = P.Print()

    def construct(self, x):
        #x = self.print_op(x, "Input shape:")
        # Forward pass
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # print('1st forward succeed')

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout1(x)
        x = self.maxpool2(x)
        # print('2nd forward succeed')

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        # print('3rd forward succeed')

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout2(x)
        x = self.maxpool4(x)
        # print('4th forward succeed')


        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.dropout3(x)
        x = self.maxpool5(x)
        # print('5th forward succeed')

        x = self.flatten(x)
        x = self.fc1(x)
        # print("Shape after fc1:", x.shape)
        x = self.relu6(x)
        x = self.dropout4(x)
        # print('6th forward succeed')

        x = self.fc2(x)
        # print("Shape after fc2:", x.shape)
        x = self.softmax(x)
        # print('last forward succeed')

        return x

if __name__ == '__main__':
    # 创建网络实例
    net = MyNet()

    # 获取网络的参数
    params = net.get_parameters()

    params_dict = {param.name: param for param in params}
    for name, param in params_dict.items():
        print(f"Layer: {name}, Parameter Shape: {param.data.shape}")
