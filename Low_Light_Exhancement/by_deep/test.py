import os
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from dataset import LowLightDataset
from models.retinex_net import RetinexNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)

# Dataset
test_dataset = LowLightDataset("LOLdataset/eval15/low", "LOLdataset/eval15/high")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 모델 로드
model = RetinexNet().to(device)
model.load_state_dict(torch.load("C:/Users/win/Desktop/by_deep/checkpoints/retinex_epoch10.pth"))
model.eval()

# 테스트
with torch.no_grad():
    for i, (low, normal) in enumerate(test_loader):
        low, normal = low.to(device), normal.to(device)
        output = model(low, normal)  # S_normal, S_low

        vutils.save_image(output, f"results/output_{i}.png")
        vutils.save_image(low, f"results/input_{i}.png")
        vutils.save_image(normal, f"results/gt_{i}.png")
