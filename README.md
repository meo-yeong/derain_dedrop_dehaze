# 사용 시 주의사항 
이 프로젝트를 실행할 때는 다음 사항들을 반드시 숙지해야 오류 없이 작동합니다.

## 실행 위치
모든 Python 스크립트는 반드시 프로젝트 최상위 폴더(Root) 에서 실행해야 합니다. 하위 폴더로 이동하여 실행할 경우, model이나 preprocessing 모듈을 찾지 못해 ModuleNotFoundError가 발생합니다.

프로젝트 루트에서 실행
python train/detrain.py
python utils/points.py

## 실행 순서
데이터셋이 준비되지 않은 상태에서 학습 스크립트를 실행하면 오류가 발생합니다. 아래 순서를 권장합니다.

데이터셋 준비 : 모델 학습 및 검증을 위해 아래의 데이터셋에서 약 5,000장의 이미지(비, 안개, 물방울 등)를 선별하여 사용하였습니다.

• 데이터셋1(https://sites.google.com/site/boyilics/website-builder/project-page) (RESIDE 관련 프로젝트 페이지)

• 데이터셋2(https://github.com/guanqiyuan/WeatherBench) (WeatherBench GitHub 저장소)


데이터 준비: utils/downloadYoloData.py 실행 혹은 사용자 데이터 준비 (data/ 폴더)

전처리 (리사이즈): preprocessing/resize.py 실행

이미지를 리사이즈하고 preprocessing/resized_image_list.txt를 생성합니다.

데이터 분할: preprocessing/split_dataset.py 실행

위에서 생성된 목록을 기반으로 dataset_split/ 폴더에 Train/Test/Val 데이터를 분할하여 저장합니다.

모델 학습:

train/detrain.py: 메인 모델 학습 (빠른 테스트용, 데이터 10% 사용)

train/detrainLite2.py: 경량화 모델 전체 학습 (데이터 100% 사용)

평가 및 추론: utils/points.py 또는 utils/pointsLite.py 실행

## 파일 및 폴더 자동 생성
학습을 진행하면 pt/ 폴더가 자동 생성되며, 그 안에 학습된 모델 가중치(.pt)가 저장됩니다.

추론을 진행하면 processedImg/ 폴더가 자동 생성되며 결과 이미지가 저장됩니다.

이 폴더들은 .gitignore에 등록되어 있어 Git에는 업로드되지 않습니다.

## 필수 라이브러리
이 프로젝트는 다음 라이브러리들을 필요로 합니다. 실행 전 설치해주세요.

pip install torch torchvision opencv-python scikit-image tqdm pillow
## 폴더 구조
```text
.
├──  .gitignore            # Git 업로드 제외 설정
├──  README.md             # 프로젝트 설명 문서
│
├──  model/                # 모델 아키텍처 정의
│   ├──  derainhaze.py     # Main 모델(DerainNet) 클래스 정의
│   ├──  lightmodel.py     # Lite 경량화 모델(DerainNetLite) 클래스 정의
│   ├──  aodNet/           # AOD-Net 실험용 노트북
│   ├──  DBaodNet/         # DB-AOD-Net 실험용 노트북
│   └──  TBaodNet/         # TB-AOD-Net 버전별 실험 기록
│
├──  preprocessing/        # 데이터 전처리 및 로더
│   ├──  prepros.py        # Custom Dataset 클래스 (RainDSSynDataset)
│   ├──  resize.py         # 이미지 리사이즈 + 파일명 목록(txt) 생성
│   ├──  resize2.py        # 이미지 리사이즈 (tqdm 진행바 포함, txt 생성 안 함)
│   ├──  split_dataset.py  # txt 목록 기반 Train(7):Test(2):Val(1) 데이터 분할
│   └──  resized_image_list.txt # resize.py 실행 시 생성되는 파일 목록
│
├──  train/                # 모델 학습 스크립트
│   ├──  detrain.py        # Main 모델 학습 (데이터 10% 사용, dedrop_derain_dehaze.pt 저장)
│   ├──  detrainLite.py    # Lite 모델 학습 (데이터 10% 사용, Litemodel.pt 저장)
│   └──  detrainLite2.py   # Lite 모델 학습 (데이터 100% 사용, Litemodel2.pt 저장)
│
└──  utils/                # 유틸리티 및 평가 도구
    ├──  points.py         # Main 모델 평가 (PSNR/SSIM 측정, dedrop_derain_dehaze.pt 로드)
    ├──  pointsLite.py     # Lite 모델 평가 (PSNR/SSIM 측정, Litemodel.pt 로드)
    ├──  testimage.py      # 단일 이미지 추론 테스트
    ├──  datatest.py       # 데이터셋 준비 1: txt 목록에 있는 이미지를 input 폴더로 복사
    ├──  datatest2.py      # 데이터셋 준비 2: 복사된 이미지 파일명 정규화

    └──  downloadYoloData.py # 외부 데이터셋(YOLO/COCO) 다운로드 도구
```

## 개발자 

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/d0gn">
        <img src="https://github.com/d0gn.png" width="80" style="border-radius:50%"><br/>
        <sub><b>d0gn</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/meo-yeong">
        <img src="https://github.com/meo-yeong.png" width="80" style="border-radius:50%"><br/>
        <sub><b>meo-yeong</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/mizzcereal">
        <img src="https://github.com/mizzcereal.png" width="80" style="border-radius:50%"><br/>
        <sub><b>mizzcereal</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/jkjnaver">
        <img src="https://github.com/jkjnaver.png" width="80" style="border-radius:50%"><br/>
        <sub><b>jkjnaver</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/sick2024471">
        <img src="https://github.com/sick2024471.png" width="80" style="border-radius:50%"><br/>
        <sub><b>sick2024471</b></sub>
      </a>
    </td>
  </tr>
</table>





