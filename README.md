# 쓰레기 사진 12종 분류 — CNN (PyTorch)

학습이 끝난 모델로 사진을 바로 분류합니다. 새로 학습할 필요는 없습니다.

## 바로 실행하기

1. `final_predict.ipynb`를 Colab(또는 Jupyter)에서 엽니다.
2. 첫 코드 셀의 `TEST_DIR`를 사진 폴더 경로로 바꿉니다. 예: `TEST_DIR = '/content/test_photos'`
3. 모든 셀을 실행합니다. Colab에서는 **런타임 → 모두 실행**입니다.

- 모델 파일 `final_model_1.pt`, `final_model_2.pt`(각 22MB)가 노트북과 같은 폴더에 없으면 GitHub에서 자동으로 내려받습니다.
- GPU가 없어도 CPU로 실행되고, 결과는 같습니다.
- 따로 설치할 패키지는 없습니다. 모두 Colab에 기본으로 들어 있습니다(torch, pillow, numpy, pandas, matplotlib, scikit-learn).
- `TEST_DIR`의 기본값은 `sample_images`입니다. 저장소 전체를 내려받았다면 경로를 바꾸지 않고 실행해 볼 수 있습니다.

## 사진 폴더 형태

```
TEST_DIR/                 TEST_DIR/
├── a.jpg                 ├── battery/
├── b.png                 │   ├── a.jpg
└── ...                   │   └── ...
                          ├── metal/
                          └── ...
→ 사진별 예측              → 사진별 예측 + 정확도 · 클래스별 점수 · 혼동행렬
```

- 하위 폴더 이름이 아래 클래스 이름과 같으면 그 폴더 이름을 정답으로 씁니다.
- 지원 형식: jpg, jpeg, png, bmp, gif, webp, tif. iPhone HEIC 사진은 jpg로 바꿔서 넣어야 합니다.
- 휴대폰 사진에 들어 있는 회전 정보(EXIF)를 반영해서 사진을 바로 세운 뒤 예측합니다.

**클래스 12종**: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass (Kaggle `mostafaabla/garbage-classification`의 폴더 이름과 같습니다)

## 결과

- **화면**
  - 사진별 예측 표: 예측 클래스, 확률, 2순위 예측
  - 정답 폴더가 있을 때: 정확도, 클래스별 점수, 혼동행렬
  - 예측 예시 그림 24장
- **파일**: `predictions.csv` (열: file, pred, prob, 2nd, 2nd prob, label, correct)

## 모델

| 항목 | 내용 |
|---|---|
| 구조 | CNN3 2개의 앙상블. CNN3 = 첫 합성곱 → residual 블록 2개씩 4단계(채널 64/128/256/512) → 전역 평균 풀링 → Dropout 0.2 → 12개 클래스. 합성곱 17층, 모델 1개당 파라미터 1,117만 개 |
| 사전학습 | **사용하지 않음**. 두 모델 모두 무작위 초기값에서 학습했습니다 |
| 학습 데이터 | 14,285장 = mostafaabla train + 외부 공개 실제 사진. 중복 제거와 라벨 정리를 했고, clothes·유리는 다양한 환경의 실제 사진으로 보강했습니다 |
| 두 모델의 차이 | 모델 1: 기본 증강 / 모델 2: 기본 증강 + 색 증강(채도 변화, 흑백 20%. 색이 곧 정답인 유리 3종에는 쓰지 않음) |
| 기본 증강 | 무작위 crop(0.35~1, 160px), 좌우·상하 반전, 밝기·대비 변화, CutMix |
| 학습 | SGD(nesterov) lr 0.1, weight decay 5e-4, warmup 5 epoch + cosine, label smoothing 0.1, 200 epoch, batch 128 |
| 예측 | 160×160 입력 → 두 모델의 확률을 평균(좌우로 뒤집은 사진도 같이 평균) → 학습 데이터의 클래스 비율로 나눠 보정(clothes 쏠림을 바로잡음) |

## 성능

| 평가 | 값 |
|---|---|
| test 정확도 (2,324장) | **0.956** (95% 범위 0.948~0.964) |
| test macro-F1 | 0.940 (0.928~0.951) |
| 평가 사이트 (AI 생성 사진 60장) | **73.33%** |
| RealWaste macro-F1 (처음 보는 실제 처리장 사진) | 0.395 |

- test는 학습과 모델 선택에 쓰지 않은 사진입니다. 괄호는 bootstrap 95% 범위입니다.
- 클래스별 F1은 clothes 0.989가 가장 높고 plastic 0.868이 가장 낮습니다. plastic과 white-glass를 metal로 잘못 보는 경우가 많습니다.
- 한계: 물체 하나를 깔끔한 배경에서 찍은 사진(학습 데이터와 비슷한 사진)에서 가장 잘 작동합니다. 배경이 복잡한 사진이나 AI 생성 사진에서는 성능이 떨어지고, 특히 trash와 paper가 약합니다.

## 파일 구성

| 파일 | 내용 |
|---|---|
| `sample_images/` | 클래스별 예시 사진 2장씩 |
| `final_predict.ipynb` | 예측 노트북 파일 |
| `final_model_1.pt`, `final_model_2.pt` | 학습된 모델 2개 : 가중치(fp16)와 클래스 이름, 입력 정규화 평균·표준편차, 입력 크기, 학습 데이터의 클래스 비율 |
| `model.py` | 평가 사이트 제출 형식의 모델 정의 (`load_model(['final_model_1.pt', 'final_model_2.pt'], device)`, `get_transform()`) |
| `presentation_02.pdf` | 발표 자료 |
| `report_02.pdf` | 종합 보고서 |

- 모델 파일을 2개로 나눈 이유: GitHub 웹 업로드 한도(25MB). 평가 사이트에는 두 모델을 합친 파일 1개를 올렸고, 예측 결과는 같습니다.

## 데이터 출처와 규칙

- 데이터(Kaggle):
  - `mostafaabla/garbage-classification`
  - `alistairking/recyclable-and-household-waste-classification`
  - `huberthamelin/garbage-classification-labels-corrections`
- PyTorch만 사용했습니다. YOLO 같은 객체 탐지·이미지 분류 패키지와 사전학습 모델은 쓰지 않았습니다.
- 평가 사이트 대비용으로 AI 생성 사진을 직접 만들어 **점검(평가)에만** 썼습니다. 학습에는 실제 사진만 썼습니다.
