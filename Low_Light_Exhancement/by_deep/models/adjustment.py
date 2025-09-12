import torch
import torch.nn as nn

class DenoiseBlock(nn.Module): 
    def __init__(self, channels): 
        super(DenoiseBlock, self).__init__() 
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1) 
        self.relu = nn.ReLU(inplace=True) 
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1) 
    def forward(self, x): 
        noise = self.conv2(self.relu(self.conv1(x))) 
        return x - noise # 입력에서 추정된 noise 제거
    
class Enhance_Net(nn.Module):
    def __init__(self, in_channels = 3, mid_channels = 64):
        super(Enhance_Net, self).__init__()

        self.conv_r1 = nn.Conv2d(in_channels+3, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv_r2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv_r3 = nn.Conv2d(mid_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.denoise = DenoiseBlock(in_channels)

        self.conv_i1 = nn.Conv2d(1, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv_i2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv_i3 = nn.Conv2d(mid_channels, 1, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, R_low, I_low):
       I_low_3 = torch.cat([I_low, I_low, I_low], dim=1)

       r = torch.cat([R_low, I_low_3], dim=1)
       r = self.relu(self.conv_r1(r))
       r = self.relu(self.conv_r2(r))
       R_hat_low = self.conv_r3(r)
       R_hat_low = self.denoise(R_hat_low)

       i = self.relu(self.conv_i1(I_low))
       i = self.relu(self.conv_i2(i))
       I_hat_low = self.conv_i3(i)     
       return R_hat_low, I_hat_low
