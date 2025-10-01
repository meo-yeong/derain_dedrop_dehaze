import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LowLightDataset
from models.retinex_net import RetinexNet
from utils import data_augmentation
import numpy as np
from torchvision.utils import save_image

# ---------------------------
# Hyperparameters & Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 100
batch_size = 4
patch_size = 64
lr = 1e-4
lambda_ir = 0.5  # gradient smoothness loss weight

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
eval_dir = "eval_results"
os.makedirs(eval_dir, exist_ok=True)

# ---------------------------
# Patch Extraction & Augmentation
# ---------------------------
def get_patch(low, normal, patch_size):
    _, H, W = low.shape
    if H < patch_size or W < patch_size:
        raise ValueError(f"이미지 크기 ({H},{W})가 patch_size ({patch_size})보다 작습니다.")
    
    x, y = np.random.randint(0, W - patch_size + 1), np.random.randint(0, H - patch_size + 1)
    low_patch = low[:, y:y+patch_size, x:x+patch_size]
    normal_patch = normal[:, y:y+patch_size, x:x+patch_size]

    mode = np.random.randint(0, 8)
    low_patch_aug = data_augmentation(low_patch.permute(1,2,0).cpu().numpy(), mode).copy()
    normal_patch_aug = data_augmentation(normal_patch.permute(1,2,0).cpu().numpy(), mode).copy()

    low_patch_aug = torch.tensor(low_patch_aug.transpose(2,0,1), dtype=torch.float32, device=low.device)
    normal_patch_aug = torch.tensor(normal_patch_aug.transpose(2,0,1), dtype=torch.float32, device=normal.device)
    return low_patch_aug, normal_patch_aug

# ---------------------------
# Dataset & DataLoader
# ---------------------------
train_dataset = LowLightDataset("LOLdataset/our485/low", "LOLdataset/our485/high")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ---------------------------
# Model & Optimizer
# ---------------------------
model = RetinexNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
l1_loss = nn.L1Loss()

# ---------------------------
# Gradient Smoothness Loss
# ---------------------------
def gradient_loss(R, I):
    grad_x = lambda t: t[:, :, :, 1:] - t[:, :, :, :-1]
    grad_y = lambda t: t[:, :, 1:, :] - t[:, :, :-1, :]

    gx_R, gy_R = grad_x(R)[:, :, :-1, :], grad_y(R)[:, :, :, :-1]
    gx_I, gy_I = grad_x(I)[:, :, :-1, :], grad_y(I)[:, :, :, :-1]

    weight = torch.exp(-gx_R - gy_R) + 1e-2
    return torch.mean(gx_I * weight) + torch.mean(gy_I * weight)

# ---------------------------
# Training Loop
# ---------------------------
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for low_batch, normal_batch in train_loader:
        low_batch = low_batch.to(device)
        normal_batch = normal_batch.to(device)

        # Patch augmentation
        low_patches, normal_patches = [], []
        for b in range(low_batch.size(0)):
            low_patch, normal_patch = get_patch(low_batch[b], normal_batch[b], patch_size)
            low_patches.append(low_patch.unsqueeze(0))
            normal_patches.append(normal_patch.unsqueeze(0))
        low_patches = torch.cat(low_patches, dim=0)
        normal_patches = torch.cat(normal_patches, dim=0)

        # ---------------------------
        # Forward
        # ---------------------------
        R_normal, I_normal, R_low, I_low = model.decom_net(normal_patches, low_patches)
        R_hat_low, I_hat_low = model.enhance_net(R_low, I_low)

        # reconstruction
        S_hat_low = R_hat_low * I_hat_low
        S_hat_low = torch.clamp(S_hat_low, 0, 1)

        # ---------------------------
        # Loss 계산
        # ---------------------------
        recon_low = R_low * I_low
        recon_normal = R_normal * I_normal
        L_decom = l1_loss(recon_low, low_patches) + l1_loss(recon_normal, normal_patches)
        L_enhance = l1_loss(S_hat_low, normal_patches)
        L_ir = lambda_ir * (gradient_loss(R_low, I_low) + gradient_loss(R_normal, I_normal))

        loss = L_decom + L_enhance + L_ir

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.6f}")

    # ---------------------------
    # Evaluation (간단 시각화)
    # ---------------------------
    model.eval()
    with torch.no_grad():
        low_sample = low_batch[0].unsqueeze(0)
        normal_sample = normal_batch[0].unsqueeze(0)
        R_normal_s, I_normal_s, R_low_s, I_low_s = model.decom_net(normal_sample, low_sample)
        R_hat_s, I_hat_s = model.enhance_net(R_low_s, I_low_s)
        S_hat_s = R_hat_s * I_hat_s
        S_hat_s = torch.clamp(S_hat_s, 0, 1)
        save_image(S_hat_s, os.path.join(eval_dir, f"epoch{epoch+1}_sample.png"))

    # ---------------------------
    # Checkpoint 저장
    # ---------------------------
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"retinex_epoch{epoch+1}.pth"))

print("[*] Training 완료.")
