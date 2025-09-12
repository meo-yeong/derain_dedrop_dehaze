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
    
#1. Original image
img = cv2.imread("input.jpg")

# BGR → YCrCb 변환
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(ycrcb)

# Y 성분만 복원
processed_Y = process_channel(Y.astype(np.float32))

# Cr, Cb는 원본 유지
ycrcb_processed = cv2.merge([processed_Y, Cr, Cb])

# 다시 BGR로 변환
photon_img_color = cv2.cvtColor(ycrcb_processed, cv2.COLOR_YCrCb2BGR)

cv2.imwrite("result.png", photon_img_color)
