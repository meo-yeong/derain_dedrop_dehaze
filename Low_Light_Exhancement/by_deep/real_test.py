import os
import torch
import torchvision.utils as vutils
import cv2
from models.retinex_net import RetinexNet

device = torch.device("cpu")
os.makedirs("results", exist_ok=True)


# 모델 로드
model = RetinexNet().to(device)
model.load_state_dict(torch.load("C:/Users/win/Desktop/by_deep/checkpoints/retinex_epoch10.pth"))
model.eval()

# 단일 이미지(real_test)
img_path = "C:/Users/win/Desktop/by_deep/input.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

# Tensor 변환 [C,H,W], 배치 차원 추가
img_tensor = torch.tensor(img, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)

# 모델 추론
with torch.no_grad():
    # 논문 구현 기준: S_normal, S_low 순서, 만약 input이 low-light라면 normal은 동일하게 사용
    output = model(img_tensor, img_tensor)  # S_normal, S_low

# 결과 저장
vutils.save_image(output, "output.jpg")
print("output.png 저장 완료!")
