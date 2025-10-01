import torch
import torch.nn as nn
import torch.nn.functional as F

class DecomNet(nn.Module):
    def __init__(self, in_channels=3, mid_channels=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels + 1, mid_channels, kernel_size, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=1, padding_mode='replicate')
        self.conv5 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=1, padding_mode='replicate')
        self.conv_out = nn.Conv2d(mid_channels, in_channels + 1, kernel_size, padding=1, padding_mode='replicate')

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        x = torch.cat((input_im, input_max), dim=1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        out = self.conv_out(x)
        R = self.sigmoid(out[:, :3, :, :])
        I = 0.5 * (torch.tanh(out[:, 3:4, :, :]) + 1.0)
        return R, I
    
class Decomposition(nn.Module):
    def __init__(self, in_channels=3, mid_channels=64):
        super(Decomposition, self).__init__()
        self.decom_net = DecomNet(in_channels, mid_channels)

    def forward(self, S_normal, S_low):
        R_normal, I_normal = self.decom_net(S_normal)
        R_low, I_low       = self.decom_net(S_low)
        return R_normal, I_normal, R_low, I_low
