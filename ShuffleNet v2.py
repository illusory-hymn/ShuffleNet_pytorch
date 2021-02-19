##  shufflenet是一个轻量级的神经网络
##  作者设计了通道切分（channel split）操作
##  在每个单元前输入通道c被划分为c-c'与c'两个分支。为避免碎片化，一个分支保持不变；
##  另一个分支遵循指南一，包含3个恒等通道数的卷积层，且1×1卷积不再分组。之后两个分支拼接为一个，通道数不变；再执行通道随机化操作。

##  ShuffleNet不仅高效，同时还很准确。原因在于：第一，提效后网络可以使用更多的通道数。
##  第二，每个单元内一半的通道直接馈入下一个单元。这可以看作是某种程度的特征再利用，类似DenseNet与CondenseNet。

##  一个epoch准确率0.9795

import torchvision
import torch.nn as nn
import torch
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math
import torch.nn.functional as F 
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import torch.utils.data as Data

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False), ## 后面有BN就不要bias
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

##   将通道数分为groups个,然后按组打乱
def channel_shuffle(x, groups):
    batchsize, n_channels, height, width = x.data.size()
    channels_per_group = n_channels // groups

    ##  reshape
    x = x.view(batchsize, groups, 
                channels_per_group, height, width)   

    x = torch.transpose(x, 1, 2).contiguous()

    ##  flatten
    x = x.view(batchsize, -1, height, width)

    return x

## benchmodel = 1实现C图操作
## benchmodel = 2实现D图操作
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            ##  assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                ##  groups原本默认值为1。groups=ouput表示每一个输出特征图由一个输入特征图产生。（群卷积）
                ##  groups = x，表示将输入分为x组，输出通道数也分为x组，每组产生oup/x个通道数。最后再concat
                nn.Conv2d(oup_inc, oup_inc, kernel_size=3, stride=stride, padding=1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, kernel_size=1, bias=False),
                nn.ReLU(inplace=True)
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, kernel_size=1, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True)
            )
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, kernel_size=1, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, kernel_size=3, stride=stride, padding=1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, kernel_size=1, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True)
            )
        
  #  @staticmethod
    def _concat(self, x, out):
        return torch.cat((x, out), 1)
    
    def forward(self, x):
        if self.benchmodel == 1:
            ##  很有意思的操作
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = self._concat(x1, self.banch2(x2))

        elif self.benchmodel == 2:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)

class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=10, input_size=224, width_mult=1.):
        super(ShuffleNetV2, self).__init__()
        assert input_size % 32 == 0
        self.stage_repeats = [4, 8, 4] ## block重复次数

        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for 
                        1x1 Grouped Convolutions""".format(num_groups))
        
        ##  building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(1, input_channel, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.features = []
        ##  building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            for i in range(numrepeat):
                if i == 0:                                  ## inp, oup, stide, benchmodel
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel
        ##  make it nn.Sequential 
        self.features = nn.Sequential(*self.features)

        ##  building last several layers 
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size/32)))

        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        x = x.view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

##  width_mult不同，中间层的通道数不同

model = ShuffleNetV2().cuda()
## 数据处理
transform = transforms.Compose([
    transforms.Resize(224), ## 224表示将图片最小的边resize到224，另一个边按相同的比例缩放
    transforms.ToTensor()
])

## parameters
LR = 0.0001
Batch_size = 32
EPOCH = 1

## data_loader
train_data = torchvision.datasets.MNIST(
    root='../mnist',  ##表示MNIST数据集下载/已存在的位置，../表示是相对于当前py文件上一级目录的mnist文件夹
    train=True,
    transform=transform,
    download=False  ## 如果没有下载就改为True自动下载
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=Batch_size, shuffle=True)
test_data = torchvision.datasets.MNIST(root='../mnist', train=False)
test_y = test_data.test_labels[:2000] ## volatile=True表示依赖这个节点的所有节点都不会进行反向求导，用于测试集
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]
test_x_1 = torch.empty(test_x.size(0), 1, 224, 224)
for i,v in enumerate(test_x):   
    temp = v[0].numpy()
    temp = Image.fromarray(temp)
    #temp = transforms.ToPILImage()(v[0])  ## 自己动手将tensor转换为Image，不要用这个函数，血的教训
    temp = transforms.Resize((224,224))(temp)
    temp = np.array(temp)
    temp = torch.Tensor(temp)/255.
    test_x_1[i][0] = temp
test_x = test_x_1.cuda()
test_x_1 = 0
test_x.volatile = True

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

## train
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        x = Variable(x).cuda()
        y = Variable(y).cuda()
        out_put = model(x)
        optimizer.zero_grad()
        loss = loss_func(out_put, y)
        loss.backward()
        optimizer.step()
        if step%100 == 0:
            print(step)
 
model.eval() 
accuracy = 0
for i, v in enumerate(test_x): ##  直接将2000张图片扔进去内存会不足
    test = v.unsqueeze(0)
    test_output = model(test)
    pred_y = torch.max(test_output, 1)[1].cpu().data.squeeze()
    accuracy += 1 if pred_y == test_y[i] else 0
accuracy /= len(test_y)
print('Epoch:', epoch, '|train loss:%.4f' % loss.item(), '|test accuracy:',accuracy)
