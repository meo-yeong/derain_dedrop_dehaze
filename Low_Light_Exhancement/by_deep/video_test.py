import os
import cv2
import torch
from models.retinex_net import RetinexNet
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_half = torch.cuda.is_available()

# --- 모델 로드 ---
model = RetinexNet().to(device)
checkpoint_path = "./checkpoints/retinex_epoch100.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
if use_half:
    model.half()

# --- 입력 영상 ---
video_path = "input_video_2.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- 출력 영상 ---
os.makedirs("results_video", exist_ok=True)
out = cv2.VideoWriter("results_video/enhanced_video.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# --- 영상 처리 ---
with torch.no_grad():
    frame_idx = 0
    max_size = 512  # optional: GPU 메모리 보호용

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR → RGB, 0~1
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        orig_h, orig_w = img.shape[:2]

        # optional: GPU 메모리 위해 resize
        scale = min(max_size / orig_h, max_size / orig_w, 1.0)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        if scale < 1.0:
            img = cv2.resize(img, (new_w, new_h))

        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
        if use_half:
            img_tensor = img_tensor.half()

        # --- RetinexNet forward (full image) ---
        # decomposition
        R, I = model.decom_net.decom_net(img_tensor)
        # enhancement
        e1 = model.enhance_net.relu(model.enhance_net.enc1(torch.cat([R, I.repeat(1,3,1,1)], dim=1)))
        e2 = model.enhance_net.relu(model.enhance_net.enc2(F.avg_pool2d(e1, 2)))
        e3 = model.enhance_net.relu(model.enhance_net.enc3(F.avg_pool2d(e2, 2)))

        d3 = F.interpolate(e3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = model.enhance_net.relu(model.enhance_net.dec3(d3))

        d2 = F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = model.enhance_net.relu(model.enhance_net.dec2(d2))

        R_hat = model.enhance_net.dec1(d2)

        # I 채널 enhancement
        i = model.enhance_net.relu(model.enhance_net.i_conv1(I))
        i = model.enhance_net.relu(model.enhance_net.i_conv2(i))
        I_hat = 0.5 * (torch.tanh(model.enhance_net.i_conv3(i)) + 1.0)

        S_hat = torch.clamp(R_hat * I_hat.repeat(1,3,1,1), 0, 1)

        # Tensor → numpy, RGB → BGR
        output_np = S_hat.squeeze(0).permute(1,2,0).cpu().numpy()
        output_np = (output_np * 255).astype(np.uint8)
        output_np = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

        # 원래 크기로 리사이즈
        if scale < 1.0:
            output_np = cv2.resize(output_np, (orig_w, orig_h))

        # 저장
        out.write(output_np)
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"[*] 처리 중: {frame_idx} 프레임")

cap.release()
out.release()
print("[*] 비디오 결과 저장 완료! results_video/ 확인")
