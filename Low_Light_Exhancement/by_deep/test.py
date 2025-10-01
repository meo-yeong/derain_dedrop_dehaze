import os
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from dataset import LowLightDataset
from models.retinex_net import RetinexNet

# ---------------------------
# 장치 설정
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results_debug", exist_ok=True)

# ---------------------------
# 테스트 데이터셋
# ---------------------------
test_dataset = LowLightDataset("LOLdataset/eval15/low", "LOLdataset/eval15/high")
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)  # batch size 자유롭게 설정

# ---------------------------
# 모델 로드
# ---------------------------
model = RetinexNet().to(device)
checkpoint_path = "./checkpoints/retinex_epoch100.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ---------------------------
# 테스트 & 결과 저장
# ---------------------------
with torch.no_grad():
    for batch_idx, (low_batch, normal_batch) in enumerate(test_loader):
        low_batch, normal_batch = low_batch.to(device), normal_batch.to(device)

        # forward (RetinexNet 내부에서 decomposition + enhancement)
        S_hat_batch = model(normal_batch, low_batch)  # 학습 시 순서와 맞추기
        S_hat_batch = torch.clamp(S_hat_batch, 0, 1)

        # 개별 이미지 저장
        for i in range(S_hat_batch.size(0)):
            global_idx = batch_idx * test_loader.batch_size + i
            vutils.save_image(S_hat_batch[i], f"results_debug/output_{global_idx}.png")
            vutils.save_image(low_batch[i], f"results_debug/input_{global_idx}.png")
            vutils.save_image(normal_batch[i], f"results_debug/gt_{global_idx}.png")
            print(f"[{global_idx}] Saved input, GT, enhanced images.")
            
print("[*] 테스트 완료. results_debug/ 폴더 확인")
