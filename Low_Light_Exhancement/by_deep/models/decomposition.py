import torch
import torch.nn as nn
import torch.nn.functional as F

class Decom_Net(nn.Module):
    def __init__(self, in_channels=3, mid_channels=64):
        super(Decom_Net, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(mid_channels, in_channels+1, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x): 
        x = self.relu(self.conv1(x)) 
        x = self.relu(self.conv2(x)) 
        x = self.sigmoid(self.conv3(x)) 
        R = x[:, :3, : :] #Reflectance(반사율) 
        I = x[:, 3:4, :, :] #Illumination(조명) 
        return R, I

class Decomposition(nn.Module):
    def __init__(self, in_channels=3, mid_channels=64):
        super(Decomposition, self).__init__()
        self.decom_net = Decom_Net(in_channels, mid_channels)
    
    def forward(self, S_normal, S_low):
        R_normal, I_normal = self.decom_net(S_normal)
        R_low, I_low = self.decom_net(S_low)
        return R_normal, I_normal, R_low, I_low
