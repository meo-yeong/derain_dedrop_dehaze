import os
import cv2
import torch
from torch.utils.data import Dataset

def get_all_images(dir_path):
    """
    dir_path 안에서 이미지 파일만 읽음 (숨김 파일 무시)
    """
    images = []
    if os.path.exists(dir_path):
        for f in os.listdir(dir_path):
            if f.startswith('.') or f.startswith('._'):  # 숨김 파일 무시
                continue
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                images.append(os.path.join(dir_path, f))
    return images

class LowLightDataset(Dataset):
    def __init__(self, low_dir, normal_dir, transform=None):
        # 모든 이미지 읽기
        low_images = get_all_images(low_dir)
        normal_images = get_all_images(normal_dir)

        # 파일 이름 기준으로 매칭
        low_dict = {os.path.basename(f): f for f in low_images}
        normal_dict = {os.path.basename(f): f for f in normal_images}

        # 공통 파일만 사용
        common_files = sorted(set(low_dict.keys()) & set(normal_dict.keys()))
        self.low_images = [low_dict[f] for f in common_files]
        self.normal_images = [normal_dict[f] for f in common_files]

        self.transform = transform

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_path = self.low_images[idx]
        normal_path = self.normal_images[idx]

        # 이미지 읽기
        low = cv2.imread(low_path)
        normal = cv2.imread(normal_path)

        # BGR -> RGB 변환 및 [0,1] 정규화
        low = cv2.cvtColor(low, cv2.COLOR_BGR2RGB) / 255.0
        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB) / 255.0

        # Tensor 변환 (C,H,W)
        low = torch.tensor(low, dtype=torch.float32).permute(2, 0, 1)
        normal = torch.tensor(normal, dtype=torch.float32).permute(2, 0, 1)

        # transform 적용 (옵션)
        if self.transform:
            low = self.transform(low)
            normal = self.transform(normal)

        return low, normal

if __name__ == "__main__":
    dataset_root = "LOLdataset"  # 데이터셋 루트 경로

    # Train 경로 수정
    low_dir = os.path.join(dataset_root, "our485/low")
    normal_dir = os.path.join(dataset_root, "our485/high")

    dataset = LowLightDataset(low_dir, normal_dir)
    print(f"총 학습 이미지 쌍: {len(dataset)}")

    # 첫 번째 이미지 확인
    low_img, high_img = dataset[0]
    print(low_img.shape, high_img.shape)
