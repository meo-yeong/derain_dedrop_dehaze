import torch
import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
import derainhaze

if __name__ == "__main__":
    print("===== 추론 스크립트 시작 =====")

    # (1) device 설정 (GPU가 없으면 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Config] Using device: {device}")

    # (2) 학습된 파일 경로 설정 (.pt 또는 .pth)
    trained_path = "epoch60+datasetplus.pt"
    trained_path2 = "160epochmidnew.pt"
    trained_path3 = "dehazer.pth"

    # 모델 로드 (이 부분은 그대로 유지)
    model = load_trained_model(trained_path, device)
    model2 = load_derainhazedrop_model(trained_path2, device)
    model3 = load_net_model(trained_path3, device)

    # (3) 추론할 이미지 경로 지정
    sample_rain_img = "./rain_storm-283.jpg"
    print(f"[Inference] 처리할 이미지: {sample_rain_img}")

    # (4) 이미지 열기 및 전처리 (이 부분은 그대로 유지)
    img_bgr = cv2.imread(sample_rain_img)
    if img_bgr is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {sample_rain_img}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 학습 때 사용한 해상도로 Resize (예: 512×1024)
    H, W = 720, 1280
    img_resized = cv2.resize(img_rgb, (W, H))   # (W, H) 순서
    img_f = img_resized.astype(np.float32) / 255.0

    # --- 여기에 모델 추론 코드를 추가해야 합니다 ---
    input_tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(device)

    # 모델 추론 실행 (어떤 모델을 사용할지는 목적에 따라 선택)
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 출력 텐서를 이미지 형태로 변환 (0~1 범위로 클리핑 후 numpy 배열로 변환)
    output_img_f = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_img_f = np.clip(output_img_f, 0.0, 1.0) # 값을 0~1 범위로 클리핑

    # --- SSIM 및 PSNR 계산 코드 추가 ---

    # 원본 이미지 (0~1 범위 float32)
    original_img_for_metrics = img_f

    # 처리된 이미지 (0~1 범위 float32)
    processed_img_for_metrics = output_img_f

    # PSNR 계산
    psnr_value = calculate_psnr(original_img_for_metrics, processed_img_for_metrics, data_range=1.0)
    print(f"계산된 PSNR: {psnr_value:.4f}")

    # SSIM 계산
    ssim_value = calculate_ssim(original_img_for_metrics, processed_img_for_metrics, data_range=1.0, channel_axis=2)
    print(f"계산된 SSIM: {ssim_value:.4f}")

    # --- 결과 이미지 저장 (선택 사항) ---
    output_img_uint8 = (output_img_f * 255).astype(np.uint8)
    output_img_bgr = cv2.cvtColor(output_img_uint8, cv2.COLOR_RGB2BGR)
    output_path = "./processed_image.jpg"
    cv2.imwrite(output_path, output_img_bgr)
    print(f"처리된 이미지 저장 완료: {output_path}")

    print("===== 추론 스크립트 종료 =====")