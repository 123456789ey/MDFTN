import torch
import torch.nn as nn
import torch.nn.functional as F
class BottleneckBlock2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BottleneckBlock2, self).__init__()
        inter_planes = out_planes * 4
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(self.relu(out))
        return torch.cat([x, out], 1)

class TransitionBlock1(nn.Module):
    def __init__(self):
        super(TransitionBlock1, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, 2)

class encode1(nn.Module):
    def __init__(self):
        super(encode1, self).__init__()
        self.conv_0 = nn.Conv2d(1,64,kernel_size=3,padding=1,stride=1)
        self.relu = nn.PReLU()
        self.res_block1 = BottleneckBlock2(64, 64)  #
        self.trans_block1 = TransitionBlock1()
        ############# Block3-down  16-16 ##############
        self.res_block2 = BottleneckBlock2(128, 128)
        self.trans_block2 = TransitionBlock1()
    def forward(self, x):
        x1 = self.relu(self.conv_0(x))
        #x1 = self.dy(x)
        x2 = self.trans_block1(self.res_block1(x1))
        #print('x2',x2.shape)
        x3 = self.trans_block2(self.res_block2(x2))
        return x1, x2, x3

class encode2(nn.Module):
    def __init__(self):
        super(encode2, self).__init__()
        self.conv_0 = nn.Conv2d(1,64,kernel_size=3,padding=1,stride=1)
        self.relu = nn.PReLU()
        self.res_block1 = BottleneckBlock2(64, 64)  #
        self.trans_block1 = TransitionBlock1()
        ############# Block3-down  16-16 ##############
        self.res_block2 = BottleneckBlock2(128, 128)
        self.trans_block2 = TransitionBlock1()
    def forward(self, x):
        x1 = self.relu(self.conv_0(x))
        x2 = self.trans_block1(self.res_block1(x1))
        x3 = self.trans_block2(self.res_block2(x2))
        return x1, x2, x3

class encode3(nn.Module):
    def __init__(self):
        super(encode3, self).__init__()
        self.conv_0 = nn.Conv2d(1,64,kernel_size=3,padding=1,stride=1)
        self.relu = nn.PReLU()
        self.res_block1 = BottleneckBlock2(64, 64)  #
        self.trans_block1 = TransitionBlock1()
        ############# Block3-down  16-16 ##############
        self.res_block2 = BottleneckBlock2(128, 128)
        self.trans_block2 = TransitionBlock1()
    def forward(self, x):
        x1 = self.relu(self.conv_0(x))
        x2 = self.trans_block1(self.res_block1(x1))
        x3 = self.trans_block2(self.res_block2(x2))
        return x1, x2, x3


# if __name__ == '__main__':
#     input1=torch.randn(20,1,128,128)    #batch_size,通道数，图片大小
#     m=encode1()
#     dict_encode = m.state_dict()
#     list_encode = list(dict_encode.keys())
#     print(len(list_encode))
#     param = get_parameter_number(m)
#     print('param=', param)
#     m1,m2,m3= m(input1)#此处有改动
#     print(m1.shape)
