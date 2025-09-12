import os
from glob import glob
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from torchvision.models import VGG16_Weights
# import net # 필요에 따라 주석 해제
# import derainhaze # 필요에 따라 주석 해제

class RainDSSynDataset(Dataset):
    def __init__(self,
                 root_gt="C:/Users/iliad/Downloads/RainDS/RainDS/RainDS_real/train_set/gt",
                 root_rain="C:/Users/iliad/Downloads/RainDS/RainDS/RainDS_real/train_set/rainstreak_raindrop",
                 img_size=(512, 1024),
                 transform=None):
        super().__init__()
        print("[Dataset] 초기화 중...")

        self.root_gt = root_gt
        self.root_rain = root_rain
        self.img_size = img_size

        # 제공해주신 "되던 코드"의 GT 패턴 목록
        gt_patterns = [
            "pie-norain-*.png",
            "norain-*.png",
            "NYU_*.jpg",
            "Aug_id_*_RandomFog_*.jpg",
            "Aug_id_*_RandomRain_*.jpg",
            "Aug_id_*_RandomRain_*.png",
            "Aug_id_*_RandomFog_*.png",
            "*_clean.png",
            "*_clean.jpg",
            "*_rain.jpg", # GT 폴더에 rain 이미지가 포함되어 있는 경우
            "*_rain.png", # GT 폴더에 rain 이미지가 포함되어 있는 경우
            "*.png" # 숫자 이름 *.png 파일을 포함하기 위해 유지
        ]
        self.gt_paths = []
        for pat in gt_patterns:
            self.gt_paths += sorted(glob(os.path.join(self.root_gt, pat)))

        self.gt_paths = sorted(list(set(self.gt_paths)))

        if len(self.gt_paths) == 0:
            raise RuntimeError(f"[Dataset] GT 이미지가 없습니다: {self.root_gt}")
        print(f"[Dataset] GT 총 이미지 개수: {len(self.gt_paths)}")

        # Rain 폴더의 모든 파일 목록 (제공해주신 코드 기준)
        self.rain_all = sorted(glob(os.path.join(self.root_rain, "*.*")))
        # 빠른 검색을 위해 Set 형태로 저장 (효율성 개선)
        self.rain_path_set = set(self.rain_all)

        if len(self.rain_all) == 0:
            raise RuntimeError(f"[Dataset] Rain 이미지가 없습니다: {self.root_rain}")
        print(f"[Dataset] Rain 총 이미지 개수: {len(self.rain_all)}")


        self.pairs = []
        # 각 파일 종류별 쌍 개수를 저장할 딕셔너리 (제공해주신 코드 기준)
        # 숫자 이름 *.png 패턴 카운트 추가
        self.pair_counts = {
            "Aug_id": 0,
            "*_clean": 0,
            "pie-norain": 0,
            "norain": 0,
            "NYU": 0,
            "numeric_png": 0, # 숫자 이름 *.png 파일 쌍 카운트 추가
            "unknown": 0 # 알 수 없는 형식의 GT 파일 (쌍이 추가되지 않음)
        }

        print("[Dataset] GT 파일과 Rain 파일 쌍 매칭 중...")
        for gt_path in self.gt_paths:
            fname = os.path.basename(gt_path)
            basename, ext = os.path.splitext(fname)
            expected_rain_fname = None
            matched_pattern_type = None

            # Case 1: *_clean.png 또는 *_clean.jpg 패턴 매칭 (순서 변경 - 앞으로 이동)
            if basename.endswith("_clean"):
                number_part = basename.replace("_clean", "")
                # GT와 동일한 확장자를 가진 _rain 파일 이름을 예상
                expected_rain_fname = f"{number_part}_rain{ext}"
                expected_rain_path = os.path.join(self.root_rain, expected_rain_fname)

                # 예상되는 Rain 파일이 Rain 폴더에 있는지 확인
                if os.path.isfile(expected_rain_path):
                     self.pairs.append((expected_rain_path, gt_path))
                     self.pair_counts["*_clean"] += 1
                else:
                     print(f"[Dataset] 경고: *_clean GT 파일 '{fname}'에 대한 대응 rain 파일 '{expected_rain_fname}'을 Rain 폴더에서 찾을 수 없습니다. 쌍 매칭에서 제외합니다.")
                     self.pair_counts["unknown"] += 1


            # Case 2: pie-norain-*.png 패턴 매칭 (순서 유지)
            elif basename.startswith("pie-norain-"):
                parts = basename.split("-")
                if len(parts) > 2 and parts[-1].replace('.png', '').isdigit():
                    idx_str = parts[-1].replace('.png', '')
                    rain_fname = f"pie-rd-rain-{idx_str}.png"
                    rain_path = os.path.join(self.root_rain, rain_fname)
                    if not os.path.isfile(rain_path):
                        raise FileNotFoundError(f"[Dataset] 대응 rain 파일을 찾을 수 없습니다: {rain_path}")
                    self.pairs.append((rain_path, gt_path))
                    self.pair_counts["pie-norain"] += 1
                else:
                    print(f"[Dataset] 경고: 'pie-norain-' 패턴과 일치하지만 형식이 예상과 다른 GT 파일, 쌍 매칭에서 제외합니다: {fname}")
                    self.pair_counts["unknown"] += 1


            # Case 3: norain-*.png 패턴 매칭 (순서 유지)
            elif basename.startswith("norain-"):
                parts = basename.split("-")
                if len(parts) > 1 and parts[-1].replace('.png', '').isdigit():
                    idx_str = parts[-1].replace('.png', '')
                    rain_fname = f"rd-rain-{idx_str}.png"
                    rain_path = os.path.join(self.root_rain, rain_fname)
                    if not os.path.isfile(rain_path):
                        raise FileNotFoundError(f"[Dataset] 대응 rain 파일을 찾을 수 없습니다: {rain_path}")
                    self.pairs.append((rain_path, gt_path))
                    self.pair_counts["norain"] += 1
                else:
                     print(f"[Dataset] 경고: 'norain-' 패턴과 일치하지만 형식이 예상과 다른 GT 파일, 쌍 매칭에서 제외합니다: {fname}")
                     self.pair_counts["unknown"] += 1


            # Case 4: NYU_*.jpg 패턴 매칭 (순서 유지)
            elif basename.startswith("NYU_"):
                prefix = basename
                pattern = os.path.join(self.root_rain, f"{prefix}_*_*.jpg")
                matched = sorted(glob(pattern))
                if len(matched) == 0:
                    raise FileNotFoundError(f"[Dataset] 대응 rain 파일을 찾을 수 없습니다: {pattern}")
                for rain_path in matched:
                    self.pairs.append((rain_path, gt_path))
                    self.pair_counts["NYU"] += 1

            # Case 5: Aug_id_... 패턴 매칭 (순서 변경 - 뒤로 이동)
            # Aug_id 패턴은 파일 이름 구조가 복잡하므로 다른 명확한 패턴들 뒤로 옮겼습니다.
            elif basename.startswith("Aug_id_") and (".png" in ext.lower() or ".jpg" in ext.lower()) and ("_RandomFog_" in basename or "_RandomRain_" in basename):
                 last_underscore_idx = basename.rfind('_')
                 if last_underscore_idx != -1:
                     prefix = basename[:last_underscore_idx]
                     # Rain 폴더에서 해당 prefix로 시작하고 .jpg 또는 .png인 파일 찾기
                     # Aug_id 패턴은 GT와 Rain의 확장자가 다를 수 있으므로 *.jpg 또는 *.png로 검색
                     pattern_jpg = os.path.join(self.root_rain, f"{prefix}_*.jpg")
                     pattern_png = os.path.join(self.root_rain, f"{prefix}_*.png")
                     matched = sorted(glob(pattern_jpg) + glob(pattern_png)) # jpg와 png 모두 검색

                     if len(matched) == 0:
                          print(f"[Dataset] 경고: Aug_id GT '{fname}'에 대한 대응 rain 파일을 찾을 수 없습니다: {pattern_jpg}, {pattern_png}")
                          self.pair_counts["unknown"] += 1
                     else:
                         for rain_path in matched:
                             self.pairs.append((rain_path, gt_path))
                             self.pair_counts["Aug_id"] += 1
                 else:
                     print(f"[Dataset] 경고: 알 수 없는 Aug_id 파일 형식, 건너뜁니다: {fname}")
                     self.pair_counts["unknown"] += 1


            # Case 6: 파일 이름이 숫자만으로 구성된 *.png 패턴 매칭 (순서 변경 - 뒤로 이동)
            # 다른 명확한 패턴들 뒤에 배치하여 먼저 처리되도록 함
            elif basename.isdigit() and ext.lower() == ".png":
                 # Rain 파일 이름은 GT 파일 이름과 동일하다고 가정
                 expected_rain_fname = fname
                 expected_rain_path = os.path.join(self.root_rain, expected_rain_fname)

                 # 예상되는 Rain 파일이 Rain 폴더에 있는지 확인
                 if os.path.isfile(expected_rain_path):
                     self.pairs.append((expected_rain_path, gt_path))
                     self.pair_counts["numeric_png"] += 1
                 else:
                     print(f"[Dataset] 경고: 숫자 이름 GT 파일 '{fname}'에 대한 대응 Rain 파일 '{expected_rain_fname}'을 Rain 폴더에서 찾을 수 없습니다. 쌍 매칭에서 제외합니다.")
                     self.pair_counts["unknown"] += 1

            else:
                # 위 모든 패턴에 해당하지 않는 GT 파일은 unknown으로 카운트
                self.pair_counts["unknown"] += 1
                print(f"[Dataset] 경고: 알려진 쌍 매칭 패턴과 일치하지 않는 GT 파일, 쌍 매칭에서 제외합니다: {fname}")


        if len(self.pairs) == 0:
             raise RuntimeError(f"[Dataset] 매칭되는 GT-Rain 쌍이 없습니다. GT 폴더: {self.root_gt}, Rain 폴더: {self.root_rain}")

        # 파일 종류별 쌍 개수 출력
        print("\n[Dataset] 매칭된 파일 종류별 쌍 개수:")
        for file_type, count in self.pair_counts.items():
            if count > 0: # 개수가 0보다 큰 경우만 출력
                 print(f"  - {file_type}: {count} 쌍")

        print(f"\n[Dataset] 총 매칭 쌍 개수: {len(self.pairs)}")
        print("[Dataset] 초기화 완료\n")


        # 이미지 크기 조정을 포함한 transform 설정 (제공해주신 코드 기준)
        if transform is None:
            print("[Dataset] 기본 transform (Resize → ToTensor) 설정")
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size[0], self.img_size[1])),
                transforms.ToTensor(),
            ])
        else:
            print("[Dataset] 사용자 지정 transform 사용")
            self.transform = transform

    def __len__(self):
        return len(self.pairs)

    # __getitem__ 메소드는 제공해주신 "되던 코드"와 동일하게 유지합니다.
    def __getitem__(self, idx):
        rain_path, gt_path = self.pairs[idx]

        try:
            # 이미지 로드 및 RGB 변환
            gt_img = Image.open(gt_path).convert("RGB")
            rain_img = Image.open(rain_path).convert("RGB")

            # transform 적용 (Resize 및 ToTensor 포함)
            gt_t = self.transform(gt_img)
            rain_t = self.transform(rain_img)

            # 딕셔너리 형태로 반환
            return {
                "rain": rain_t,
                "gt": gt_t,
                "rain_path": rain_path,
                "gt_path": gt_path
            }
        except FileNotFoundError as e:
             print(f"[Dataset] 오류: 파일을 찾을 수 없습니다 - {e}")
             return None
        except Exception as e:
             print(f"[Dataset] 오류: 이미지 로딩 또는 처리 중 오류 발생 ({os.path.basename(rain_path)}, {os.path.basename(gt_path)}) - {e}")
             return None

# # 테스트를 위한 예시 코드 (필요시 주석 해제 후 사용)
# if __name__ == '__main__':
#     try:
#         basic_transform = transforms.Compose([
#             transforms.ToTensor()
#         ])

#         dataset = RainDSSynDataset(
#             root_gt="C:/Users/iliad/Downloads/RainDS/RainDS/RainDS_real/train_set/gt",
#             root_rain="C:/Users/iliad/Downloads/RainDS/RainDS/RainDS_real/train_set/rainstreak_raindrop",
#             img_size=(512, 512) # 예시 크기
#             # transform=None # 기본 transform 사용
#         )
#         print(f"데이터셋 크기: {len(dataset)}")
#         if len(dataset) > 0:
#             loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0) # num_workers는 환경에 맞게 조절
#             for i, batch in enumerate(loader):
#                 if batch is None:
#                     print(f"Skipping batch {i} due to None item")
#                     continue
#                 print(f"Batch {i}:")
#                 print(f"  'rain' tensor shape: {batch['rain'].shape}")
#                 print(f"  'gt' tensor shape: {batch['gt'].shape}")
#                 print(f"  'rain_path' example: {batch['rain_path'][0]}")
#                 print(f"  'gt_path' example: {batch['gt_path'][0]}")
#                 if i >= 2:
#                     break

#     except RuntimeError as e:
#         print(f"데이터셋 초기화 실패: {e}")
#     except Exception as e:
#         print(f"예상치 못한 오류 발생: {e}")