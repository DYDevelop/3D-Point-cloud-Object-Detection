# MMdetection3D를 이용한 실습
-----------------------------------------
## week. 7
환경설정 구축
1. 4명(백근주, 김대식, 박민배, 박준서) 는 Colab pro 환경에서, 1명(김동영) 은 Desktop 환경에서 진행하기로 결정함
2. 라이브러리는 MMdetection3d를 사용하기로 결정함
3. 기본적인 Config 파일 구성 및 실제 내부 인자에 대해 공부함.
------------------------------------------------
## week. 8
1. Baseline은 MMdetection3d의 pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py를 Baseline으로 정함
2. 해당 github에 저장되어있는 pre-train model이 아닌 Colab pro를 이용하여 직접 학습시킨 모델을 baseline으로 설정함

------------------------------------------------
## week. 9 ~ 14
1. 각자 Baseline이 잘 동작했는지 여부 확인
2. 각자 Baseline보다 성능을 어떻게 더 높일지 근거(논문, 실험표)를 토대로 계획 세움
------------------------------------------------
## 모델 개선 방법

| 주차 | 개선 방법 | 관련 자료 링크 |
| :--: | :------: | :------------: |
| 9 | PointPillar의 backbone에 따른 성능 및 속도 차이 관련 논문 → CSPdarknet으로 성능 향상 확인 | [Paper](https://arxiv.org/pdf/2209.15252.pdf) |
| 11 | CSPDarknet과 CSPBepBackbone로 backbone 변경 | [Paper](https://arxiv.org/pdf/2004.10934.pdf) |
| 12 | CSPBepBackbone의 구조 변경를 변경해 연산량(GFlops)를 늘려 성능향상을 목표  | [Paper](https://arxiv.org/pdf/2209.02976.pdf) |
| 14 | ConvNeXt_tiny와   ConvNeXt_small로 backbone 변경 | [Paper](https://arxiv.org/pdf/2201.03545.pdf) |

----------------------------------------------------------------
# 성능표
| Week |         Model           | plane information |Optimizer | Batch size | Learning rate | Scheduler | Epochs | Overall 3D AP@40 | Easy | Moderate | Hard | Model | FLOPs | Model_depths |
| :---: | :-------------: | :--:|:--------:| :---------:| :-----------: | :-------: | :----: | :--------------: | :--: | :------: | :--: | :--: | :--: | :--------: |
| 8 | Top Down (Baseline) | X |AdamW | 6 | 0.001 | CosineAnnealingLR | 160 | | 73.9592 | 62.3412 | 58.3332 | model/pointpillars.py | 29.71 | [3, 5, 5]
| 9 | CSPDarkNet | X | AdamW | 6 | 0.001 | CosineAnnealingLR | 160 | | 74.4968 | 63.0843 | 59.1675 | model/pointpillars.py | 41.97 | [3, 6, 9]
| 9 | CSPBepNet | X | AdamW | 6 | 0.001 | CosineAnnealingLR | 160 | | 72.5008 | 61.0834 | 56.8824 | model/pointpillars.py | 20.28 | [3, 6, 9]
| 12 | CSPBepNet | X | AdamW | 6 | 0.001 | CosineAnnealingLR | 160 | | 72.7452| 60.5919 | 56.6782 | model/pointpillars.py |35.45 | [1, 4, 8, 20]
| 12 | CSPBepNet | X | AdamW | 6 | 0.001 | CosineAnnealingLR | 160 | | 73.5264 | 60.9541 | 57.1894 | model/pointpillars.py |46.02 | [1, 6, 12, 12]
| 14 | ConvNeXt_tiny | X | AdamW | 6 | 0.001 | CosineAnnealingLR | 160 | | 70.4475 | 58.1876 | 54.4878 | model/pointpillars.py | 29.62 | [3, 9, 3]
| 14 | ConvNeXt_small | X | AdamW | 6 | 0.001 | CosineAnnealingLR | 160 | | 68.4511 | 57.3610 | 53.4336 | model/pointpillars.py | 62.88 | [3, 27, 3]
