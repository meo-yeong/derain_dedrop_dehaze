import torch
import torch.nn as nn
from models.decomposition import Decomposition
from models.adjustment import Enhance_Net


class RetinexNet(nn.Module):
    def __init__(self):
        super(RetinexNet, self).__init__()
        self.decom_net = Decomposition()
        self.enhance_net = Enhance_Net()
    
    def forward(self, S_normal, S_low):
        # 두 입력을 동시에 넘겨주기
        R_normal, I_normal, R_low, I_low = self.decom_net(S_normal, S_low)
    
        R_hat_low, I_hat_low = self.enhance_net(R_low, I_low)
    
        I_hat_low_3 = torch.cat([I_hat_low, I_hat_low, I_hat_low], dim=1)
        S_hat_low = R_hat_low * I_hat_low_3

        return S_hat_low
