import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.retinex_net import RetinexNet
from torch.utils.data import DataLoader
from dataset import LowLightDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset 불러오기
train_dataset = LowLightDataset("LOLdataset/our485/low", "LOLdataset/our485/high")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 모델, 손실, 옵티마이저
model = RetinexNet().to(device)
criterion = nn.L1Loss()   # L1 loss (픽셀 차이)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

checkpoint_dir = "C:/Users/win/Desktop/by_deep/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# 학습 루프
for epoch in range(10):
    model.train()
    epoch_loss = 0
    for low, normal in train_loader:
        low, normal = low.to(device), normal.to(device)
        output = model(low, normal)

        loss = criterion(output, normal)  # GT normal-light 이미지와 비교
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")

    # 체크포인트 저장
    torch.save(model.state_dict(), f"checkpoints/retinex_epoch{epoch+1}.pth")
