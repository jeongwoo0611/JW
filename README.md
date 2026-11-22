# 쓰레기 사진 12종 분류 — 직접 설계한 CNN (PyTorch)

학습이 끝난 모델로 사진을 바로 분류합니다. 새로 학습할 필요는 없습니다.

## 바로 실행하기

1. `final_predict.ipynb`를 Colab(또는 Jupyter)에서 엽니다. 
2. 첫 코드 셀의 `TEST_DIR`를 사진 폴더 경로로 바꿉니다. 예: `TEST_DIR = '/content/test_photos'`
3. 모든 셀을 실행합니다. Colab에서는 **런타임 → 모두 실행**입니다.

- 모델 파일 `final_model.pt`(22MB)가 노트북과 같은 폴더에 없으면 GitHub에서 자동으로 내려받습니다.
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
| 구조 | CNN3: 첫 합성곱 → residual 블록 2개씩 4단계(채널 64/128/256/512) → 전역 평균 풀링 → Dropout 0.2 → 12개 클래스. 합성곱 17층, 파라미터 1,117만 개 |
| 사전학습 | **사용하지 않음**. 무작위 초기값에서 학습했습니다 |
| 입력 | 192×192 RGB. 학습 때는 160px로 무작위 crop하고, 평가 크기는 val로 골랐습니다 |
| 학습 데이터 | 14,039장 = mostafaabla train 10,821장 + 외부 공개 데이터 3,218장. 중복 제거와 라벨 정리를 했습니다 |
| 증강 | 무작위 crop(0.35~1), 좌우·상하 반전, 밝기·대비 변화(유리 색이 클래스라 색상 변화는 쓰지 않음), CutMix |
| 학습 | SGD(nesterov) lr 0.1, weight decay 5e-4, warmup 5 epoch + cosine, label smoothing 0.1, 200 epoch, batch 128 |
| 선택 | 같은 설정으로 seed 3개를 학습하고, val macro-F1이 가장 높은 seed를 골랐습니다 |

## 성능

test 2,324장으로 측정했습니다. 학습과 모델 선택에 쓰지 않은 사진이고, 괄호는 bootstrap 95% 범위입니다.

| 지표 | 값 |
|---|---|
| 정확도 | **0.954** (0.945~0.963) |
| macro-F1 | 0.939 (0.926~0.950) |
| battery recall | 0.915 (0.864~0.958) |

- 클래스별 F1은 clothes 0.992가 가장 높고 metal 0.874가 가장 낮습니다. plastic과 white-glass를 metal로, 흔하지 않은 모양의 물체를 shoes로 잘못 보는 경우가 많습니다.
- 한계: 물체 하나를 깔끔한 배경에서 찍은 사진(학습 데이터와 비슷한 사진)에서 잘 작동합니다. 실제 처리장처럼 배경이 복잡하거나 여러 물체가 섞인 사진에서는 성능이 크게 떨어집니다. 외부 데이터 RealWaste에서 macro-F1은 약 0.30이었습니다.

## 파일 구성

| 파일 | 내용 |
|---|---|
| `sample_images/` | 클래스별 예시 사진 2장씩 |
| `final_predict.ipynb` | 예측 노트북 파일 |
| `final_model.pt` | 학습된 모델 : 가중치(fp16)와 클래스 이름, 입력 정규화 평균·표준편차, 입력 크기 |
| `presentation_01.pdf` | 발표 자료 |
| `report_01.pdf` | 종합 보고서 |

## 데이터 출처와 규칙

- 데이터(Kaggle):
  - `mostafaabla/garbage-classification`
  - `alistairking/recyclable-and-household-waste-classification`
  - `huberthamelin/garbage-classification-labels-corrections`
- PyTorch만 사용했습니다. YOLO 같은 객체 탐지·이미지 분류 패키지와 사전학습 모델은 쓰지 않았습니다.
