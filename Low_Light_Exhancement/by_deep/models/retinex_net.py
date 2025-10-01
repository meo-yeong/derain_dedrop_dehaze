import torch
import torch.nn as nn
from models.decomposition import Decomposition
from models.adjustment import UNetEnhance

class RetinexNet(nn.Module):
    def __init__(self):
        super(RetinexNet, self).__init__()
        self.decom_net = Decomposition()
        self.enhance_net = UNetEnhance()
    
    def forward(self, S_normal, S_low):
        R_normal, I_normal, R_low, I_low = self.decom_net(S_normal, S_low)
        R_hat_low, I_hat_low = self.enhance_net(R_low, I_low)
        
        S_hat_low = R_hat_low * I_hat_low  # broadcasting으로 곱
        S_hat_low = torch.clamp(S_hat_low, 0, 1)
        return S_hat_low

