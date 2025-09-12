# derain_dedrop_dehaze

## derainhaze.py
모델 구조가 적혀있는 파일 모델의 내부구조를 변경하고싶을때 들어가서 수정.

## prepros.py
데이터셋 전처리 파일 detrain.py에서 데이터(사진)을 불러올때 데이터 인식해서 불러오게하는 파일 
    > root_gt= 클린한 이미지 (정답 이미지) 파일주소
    > root_rain= 헤이즈 물방울 비오는 이미지 (처리할 이미지) 파일주소
    > gt_patterns[ 이안에 인식할 사진의 이름들을 기입.]
## detrain.py
prepros에서 불러온 데이터를 derainhaze의 모델로 학습하는 파일
    > root_gt_folder = 클린한 이미지 (정답 이미지) 파일주소
    > root_rain_folder = 헤이즈 물방울 비오는 이미지 (처리할 이미지) 파일주소
    > val_gt_folder   = 검증용 클린한 이미지 (정답 이미지) 파일주소
    > val_rain_folder = 검증용 헤이즈 물방울 비오는 이미지 (처리할 이미지) 파일주소
## testimage.py
모델 불러와서 처리해보는 파일
    > MODEL_WEIGHTS_PATH = 학습된 모델 가중치 파일 경로
    > INPUT_IMAGE_PATH = 비를 제거할 입력 이미지 경로
    > OUTPUT_IMAGE_PATH = 결과 이미지를 저장할 경로
## points.py
ssim psnr 점수 확인하는 파일 <<원본과 비교하기때문에 수정 약간필요
    > trained_path = pt파일 불러오기
    > sample_rain_img = haze한 이미지 경로
    > output_path = 저장경로.jpg로 저장








