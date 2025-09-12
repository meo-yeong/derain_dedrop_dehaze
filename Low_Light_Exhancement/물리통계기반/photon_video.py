import numpy as np
import cv2
from scipy.special import j1

def process_channel(channel, Np = 1000):
    #2. Nomalized irradiance
    img_norm = channel / channel.max()

    #4. Photon counting model
    photon_counts = np.random.poisson(Np * img_norm)

    # map 계산전 λ_x 초기화
    lambda_x = photon_counts.copy()

    #5. low-photon
    lambda_u = np.fft.fft2(lambda_x/lambda_x.sum())

    #Hu 정의
    rows, cols = photon_counts.shape
    crow, ccol = rows//2, cols//2
    R = 10
    Hu = np.ones((rows, cols))
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x-ccol)**2 + (y-crow)**2 <= R**2
    Hu[mask_area] = 0

    lambda_u_filtered = Hu * lambda_u
    correction = np.fft.ifft2(lambda_u_filtered).real

    #6. MAP
    avg = np.mean(lambda_x)
    std = np.std(lambda_x)
    a = avg ** 2 / std ** 2
    b = avg / std ** 2

    lambda_x_hat = (photon_counts + a) / (Np * (1+b))

    #7. wavlet
    lam = 532e-9
    w = 1e-3
    z = 1.0
    k = 2 * np.pi / lam
    x_grid = np.linspace(-1e-3, 1e-3, cols)
    y_grid = np.linspace(-1e-3, 1e-3, rows)
    X, Y = np.meshgrid(x_grid, y_grid)
    r = np.sqrt(X**2 + Y**2)

    A = np.pi * w**2

    krz = (k * w * r) / z
    U = np.exp(1j * k * z) * np.exp(1j * k * r**2 / (2*z)) * (A / (1j * lam * z)) * (2 * j1(krz) / (krz + 1e-12))

    C = 1/Np * np.fft.ifft2(np.fft.fft2(U) * np.fft.fft2(photon_counts)).real
    C_scaled = C / C.max() * lambda_x_hat.max()
    
    #8. Photon counting image(nomalize for display)
    alpha = 0.05  # 0~1 사이에서 실험
    final_img = (1 - alpha) * lambda_x_hat + alpha * C_scaled
    final_img = np.clip(final_img, 0, None)
    final_img_uint8 = (final_img / final_img.max() * 255).astype(np.uint8)
    return final_img_uint8
    
# ---------------- 영상 처리 ----------------
input_video = "input.mp4"  # 입력 영상
output_video = "night_processed.mp4"

cap = cv2.VideoCapture(input_video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(output_video,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 채널별 처리
    channels = cv2.split(frame.astype(np.float32))
    processed_channels = [process_channel(ch) for ch in channels]

    # 통합 최대값 기준 정규화 + uint8
    max_val = max(ch.max() for ch in processed_channels)
    processed_channels = [(ch / max_val * 255).astype(np.uint8) for ch in processed_channels]

    # 프레임 합치기
    processed_frame = cv2.merge(processed_channels)  # BGR 3채널
    out.write(processed_frame)
