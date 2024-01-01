from torch import nn
import torch

class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(4096, 128)
        self.linear2 = nn.Linear(128, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(type(x))
        x = self.conv1(x)
        # print(x.shape)
        x = self.maxpool1(x)
        # print(x.shape)

        x = self.conv2(x)
        # print(x.shape)
        x = self.maxpool2(x)
        # print(x.shape)

        x = self.conv3(x)
        # print(x.shape)
        x = self.maxpool3(x)
        # print(x.shape)

        x = self.conv4(x)
        # print(x.shape)
        x = self.maxpool4(x)
        # print(x.shape)

        x = self.conv5(x)
        # print(x.shape)
        x = self.maxpool5(x)
        # print(x.shape)

        x = self.flatten(x)
        # print(x.shape)
        x = self.linear1(x)

        x = self.linear2(x)
        x = self.softmax(x)

        return x
if __name__ == '__main__':
    x = torch.randn(1000, 1, 150, 150)
    myModel = MyModel()
    out = myModel(x)
    print(out)