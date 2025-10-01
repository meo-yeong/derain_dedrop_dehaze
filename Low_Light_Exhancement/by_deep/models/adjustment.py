import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetEnhance(nn.Module):
    def __init__(self, in_channels=6, base_channels=64):  # R_low + I_low_3 => 6채널
        super().__init__()
        # --- Encoder ---
        self.enc1 = nn.Conv2d(in_channels, base_channels, 3, 1, 1)        # 6 -> 64
        self.enc2 = nn.Conv2d(base_channels, base_channels*2, 3, 1, 1)    # 64 -> 128
        self.enc3 = nn.Conv2d(base_channels*2, base_channels*4, 3, 1, 1)  # 128 -> 256

        # --- Decoder ---
        # concat 방식으로 skip connection -> 채널 수 = decoder_input_channels + skip_channels
        self.dec3 = nn.Conv2d(base_channels*4 + base_channels*2, base_channels*2, 3, 1, 1)  # 256+128 -> 128
        self.dec2 = nn.Conv2d(base_channels*2 + base_channels, base_channels, 3, 1, 1)      # 128+64 -> 64
        self.dec1 = nn.Conv2d(base_channels, 3, 3, 1, 1)  # R_hat 출력

        self.relu = nn.ReLU(inplace=True)

        # I 채널 처리용
        self.i_conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.i_conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.i_conv3 = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, R_low, I_low):
        # I 채널 3채널로 확장
        I_low_3 = I_low.repeat(1, 3, 1, 1)
        x = torch.cat([R_low, I_low_3], dim=1)

        # --- Encoder ---
        e1 = self.relu(self.enc1(x))                          # B,64,H,W
        e2 = self.relu(self.enc2(F.avg_pool2d(e1, 2)))       # B,128,H/2,W/2
        e3 = self.relu(self.enc3(F.avg_pool2d(e2, 2)))       # B,256,H/4,W/4

        # --- Decoder (concat skip connection) ---
        d3 = F.interpolate(e3, scale_factor=2, mode='bilinear', align_corners=False)  # H/2,W/2
        d3 = torch.cat([d3, e2], dim=1)                      # 채널 256+128=384
        d3 = self.relu(self.dec3(d3))                        # 384->128

        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)  # H,W
        d2 = torch.cat([d2, e1], dim=1)                      # 128+64=192
        d2 = self.relu(self.dec2(d2))                        # 192->64

        R_hat = self.dec1(d2)                                # 64->3

        # --- I 채널 ---
        i = self.relu(self.i_conv1(I_low))
        i = self.relu(self.i_conv2(i))
        I_hat = 0.5 * (torch.tanh(self.i_conv3(i)) + 1.0)

        return R_hat, I_hat
