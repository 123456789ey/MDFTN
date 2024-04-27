import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_onedown import encode1,encode2,encode3
class rmse(nn.Module):
    def __init__(self,depth,mid_channels=64):
        super(rmse,self).__init__()
        self.resblock = nn.Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1,stride=1)
        self.sig = nn.Sigmoid()
        self.conv =nn.Sequential(nn.Conv2d(mid_channels,mid_channels,kernel_size=1,padding=0,stride=1),nn.PReLU(mid_channels),
                                 nn.Conv2d(mid_channels,mid_channels,kernel_size=1,padding=0,stride=1))
        self.depth = str(depth)
    def forward(self,x):
        output = []
        output.append(x)
        size = len(self.depth)
        for i in range(size):
            out1 = self.resblock(output[i])
            out = nn.AdaptiveAvgPool2d((1,1))(out1)
            out = self.conv(out)
            out = self.sig(out)
            out = torch.mul(out, out1)
            out = out + output[(i)]
            output.append(out)
        return x + output[(size-1)]

#通道注意力和空间注意
class channel_spatial_attention(nn.Module):
    def __init__(self,in_channel=64,kerner_size=3):
        super(channel_spatial_attention,self).__init__()
        self.kernel_size=kerner_size
        # channel_attention
        self.conv_sp = nn.Conv2d(in_channel,in_channel,kernel_size=kerner_size,  stride=1,padding=1,dilation=1,groups=in_channel) #groups是对所有通道的数分成groups组进行通道卷积，进行的卷积最后cat连接
        #spatial_attention
        self.ave_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_ch =nn.Conv1d(1,1,kernel_size=kerner_size,padding=(kerner_size-1)//2)
        self.conv = nn.Conv2d(in_channel, in_channel*kerner_size*kerner_size ,kernel_size=1,padding=0)
        self.unfold = nn.Unfold(kernel_size=kerner_size,dilation=1,padding=1,stride=1)
    def forward(self,x):
        b,c,h,w = x.size()   #[20, 64, 128, 128]
        channel = self.conv_sp(x)   # channel_attention    #[20, 64, 128, 128]
        #spatial_attention
        pool = self.ave_pool(x)   #[20, 64, 1, 1]
        spatial =self.conv_ch(pool.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)   #transpose进行交换，squeeze减小维度，unsqueeze增加维度
        channel_spatial =channel + spatial   #[20, 64, 128, 128]
        out_1 = self.conv(channel_spatial)    #[20, 576, 128, 128]
        filter_x = out_1.reshape([b,c,self.kernel_size*self.kernel_size,h,w])     # #[20, 64, 9, 128, 128]
        unfold_x=self.unfold(x).reshape(b,c,-1,h,w)   #[20, 64, 9, 128, 128]
        out_2 = (filter_x*unfold_x).sum(2)   #[20, 64, 128, 128]
        return out_2
#************************************上采样**************************
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self):
        super(Up,self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)  # 进行融合裁剪
        return x

class decode1(nn.Module):
    def __init__(self):
        super(decode1,self).__init__()
        self.up = Up()
        self.chsp2 = channel_spatial_attention(128, 3)
        self.chsp1 = channel_spatial_attention(64, 3)
        self.conv2 = nn.Conv2d(256+128, 128, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1,kernel_size=3,padding=1)
        self.relu = nn.PReLU()
    def forward(self,x1,x2,x3):
        #x2 = self.chsp2(x2)
        x1_1 = self.relu(self.conv2(self.up(x3, x2)))
        #x1 = self.chsp1(x1)
        x1_1 = self.chsp2(x1_1)
        x0_1 = self.relu(self.conv1(self.up(x1_1, x1)))
        x0_1 = self.chsp1(x0_1)
        out = self.conv_out(x0_1)
        return out
class decode2(nn.Module):
    def __init__(self):
        super(decode2,self).__init__()
        self.up = Up()
        self.chsp2 = channel_spatial_attention(128, 3)
        self.chsp1 = channel_spatial_attention(64, 3)
        self.conv2 = nn.Conv2d(256+128, 128, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1,kernel_size=3,padding=1)
        self.relu = nn.PReLU()
    def forward(self,x1,x2,x3):
        #x2 = self.chsp2(x2)
        x1_1 = self.relu(self.conv2(self.up(x3, x2)))
        #x1 = self.chsp1(x1)
        x1_1 = self.chsp2(x1_1)
        x0_1 = self.relu(self.conv1(self.up(x1_1, x1)))
        x0_1 = self.chsp1(x0_1)
        out = self.conv_out(x0_1)
        return out
class decode3(nn.Module):
    def __init__(self):
        super(decode3,self).__init__()
        self.up = Up()
        self.chsp2 = channel_spatial_attention(128, 3)
        self.chsp1 = channel_spatial_attention(64, 3)
        self.conv2 = nn.Conv2d(256+128, 128, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1,kernel_size=3,padding=1)
        self.relu = nn.PReLU()
    def forward(self,x1,x2,x3):
        #x2 = self.chsp2(x2)
        x1_1 = self.relu(self.conv2(self.up(x3, x2)))
        #x1 = self.chsp1(x1)
        x1_1 = self.chsp2(x1_1)
        x0_1 = self.relu(self.conv1(self.up(x1_1, x1)))
        x0_1 = self.chsp1(x0_1)
        out = self.conv_out(x0_1)
        return out

class MSDnet(nn.Module):
    def __init__(self):
        super(MSDnet,self).__init__()
        self.E1 = encode1()
        self.E2 = encode2()
        self.E3 = encode3()
        self.rmse = rmse(4,256)
        self.conv = nn.Conv2d(256*3,256,kernel_size=3,padding=1,stride=1)
        self.D1 = decode1()
        self.D2 = decode2()
        self.D3 = decode3()
    def forward(self,x1,x2,x3):
        x11,x12,x13 = self.E1(x1)
        x21,x22,x23 = self.E2(x2)
        x31,x32,x33 = self.E3(x3)
        cat = self.conv(torch.cat([x13,x23,x33],dim=1))
        res = self.rmse(cat)
        #out1,out2,out3 = torch.split(res,(256,256,256),dim=1)
        out1 = self.D1(x11,x12,res)
        out2 = self.D2(x21,x22,res)
        out3 = self.D3(x31,x32,res)
        return out1,out2,out3


# if __name__ == '__main__':
#     input1=torch.randn(20,1,128,128)    #batch_size,通道数，图片大小
#     input2=torch.randn(20,1,128,128)    #batch_size,通道数，图片大小
#     input3=torch.randn(20,1,128,128)    #batch_size,通道数，图片大小
#     m=MSDnet()
#     m1,m2,m3= m(input1,input2,input3)#此处有改动
#     print(m1.shape,m2.shape,m3.shape)
