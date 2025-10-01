import os
import torch
import torchvision.utils as vutils
import cv2
from models.retinex_net import RetinexNet
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)

# --- 모델 로드 ---
model = RetinexNet().to(device)
checkpoint_path = "./checkpoints/retinex_epoch100.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# --- Single image forward 함수 ---
def forward_single(model, img_tensor):
    with torch.no_grad():
        # decomposition
        R, I = model.decom_net.decom_net(img_tensor)

        # enhancement (skip connection 크기 맞춤)
        # Encoder feature sizes
        e1 = model.enhance_net.relu(model.enhance_net.enc1(torch.cat([R, I.repeat(1,3,1,1)], dim=1)))
        e2 = model.enhance_net.relu(model.enhance_net.enc2(F.avg_pool2d(e1, 2)))
        e3 = model.enhance_net.relu(model.enhance_net.enc3(F.avg_pool2d(e2, 2)))

        # Decoder
        d3 = F.interpolate(e3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = model.enhance_net.relu(model.enhance_net.dec3(d3))

        d2 = F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = model.enhance_net.relu(model.enhance_net.dec2(d2))

        R_hat = model.enhance_net.dec1(d2)

        # I 채널
        i = model.enhance_net.relu(model.enhance_net.i_conv1(I))
        i = model.enhance_net.relu(model.enhance_net.i_conv2(i))
        I_hat = 0.5 * (torch.tanh(model.enhance_net.i_conv3(i)) + 1.0)

        S_hat = torch.clamp(R_hat * I_hat.repeat(1,3,1,1), 0, 1)

    return S_hat

# --- 이미지 불러오기 ---
img_path = "input.jpg"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")

# optional: GPU 메모리 위해 최대 크기 제한
max_size = 512
orig_h, orig_w = img.shape[:2]
scale = min(max_size / orig_h, max_size / orig_w, 1.0)
new_h, new_w = int(orig_h * scale), int(orig_w * scale)
img = cv2.resize(img, (new_w, new_h))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
img_tensor = torch.tensor(img, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)

# --- 추론 ---
output = forward_single(model, img_tensor)

# --- 결과 저장 ---
vutils.save_image(output, "results/enhanced_output.png")
vutils.save_image(img_tensor, "results/input.png")

print("[*] 결과 저장 완료! results/enhanced_output.png 확인")
