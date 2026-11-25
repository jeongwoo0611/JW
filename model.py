"""쓰레기 사진 12종 분류 — 최종 모델 정의 (평가 사이트 73.33% 제출과 같은 모델·전처리)

모델: CNN3 2개의 앙상블 — torch.nn으로 직접 정의, 사전학습 가중치 사용 안 함
      평가 크기 160px + 좌우 뒤집기 평균 + 클래스 비율 보정
가중치: final_model_1.pt (CNN3 #1, 기본 증강) + final_model_2.pt (CNN3 #2, 색 증강)
      GitHub 웹 업로드 한도(25MB) 때문에 모델별로 나눠 올림 → load_model에 두 파일을 목록으로 넘기면 합쳐서 앙상블로 만듦
      평가 사이트에는 두 모델을 합친 파일 1개를 올렸음 (그 파일도 그대로 불러옴)
      가중치 파일 = {'states': 모델별 가중치 목록, 'flip': 좌우 뒤집기 평균 여부, 'prior': 클래스 비율 보정에 쓰는 학습 데이터 비율,
                     'members', 'classes', 'mean', 'std', 'image_size', 'model'}
평가 서버가 쓰는 것: load_model(weights_path, device), get_transform(), CLASSES
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

# 모델 출력 인덱스 순서 (가중치 파일의 'classes'와 같음 = 알파벳 순)
CLASSES = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass',
]

# 평가 입력 크기와 정규화 평균·표준편차 (가중치 파일의 'image_size', 'mean', 'std'와 같아야 함)
IMG_SIZE = 160
MEAN = [0.6614172458648682, 0.6191232800483704, 0.5877915620803833]
STD = [0.28159111738204956, 0.29277026653289795, 0.30730533599853516]


# ------------------------------------------------------------ 1. 모델 구조 (학습 때와 같음)

# residual 블록: 3x3 합성곱 2개를 거친 값 + 입력을 그대로 더함 (크기가 바뀌는 블록은 1x1 합성곱으로 입력 크기를 맞춤)
class Block(nn.Module) :
    def __init__(self, c_in, c_out, stride) :
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride, 1, bias = False), nn.BatchNorm2d(c_out), nn.ReLU(inplace = True),
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias = False), nn.BatchNorm2d(c_out))
        if stride == 1 and c_in == c_out :
            self.skip = nn.Identity()
        else :
            self.skip = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias = False), nn.BatchNorm2d(c_out))

    def forward(self, x) :
        return F.relu(self.body(x) + self.skip(x))


# CNN3: 첫 합성곱(크기 1/2) → residual 블록 2개씩 4단계(채널 64/128/256/512) → 전역 평균 풀링 → Dropout → 12개 클래스 점수
class CNN3(nn.Module) :
    def __init__(self, widths = (64, 128, 256, 512), dropout = 0.2, n_classes = 12) :
        super().__init__()
        layers = [nn.Conv2d(3, widths[0], 3, 2, 1, bias = False), nn.BatchNorm2d(widths[0]), nn.ReLU(inplace = True)]
        c = widths[0]
        for i, w in enumerate(widths) :
            layers += [Block(c, w, 1 if i == 0 else 2), Block(w, w, 1)]
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, n_classes))

    def forward(self, x) :
        return self.head(self.features(x))      # softmax 전 logits, 모양 (1, 12)


# 앙상블: 같은 구조의 CNN3 여러 개의 확률을 평균함 (flip=True면 좌우로 뒤집은 사진의 확률도 같이 평균함)
#  - prior가 있으면 클래스 비율 보정: 평균 확률을 학습 데이터 클래스 비율로 나눔 (log에서는 빼기)
#    → 평가 사진은 클래스마다 같은 수라서, 학습 데이터에 많은 클래스 쪽으로 쏠린 예측을 바로잡음 (베이즈 규칙)
#  - 출력 = log(평균 확률) - log(학습 비율) → softmax를 하면 보정한 확률이 나오므로 logits로 쓸 수 있음, 모양 (1, 12)
class Ensemble(nn.Module) :
    def __init__(self, n_models, flip, prior = None) :
        super().__init__()
        self.models = nn.ModuleList([CNN3(n_classes = len(CLASSES)) for _ in range(n_models)])
        self.flip = flip
        log_prior = torch.log(torch.tensor(prior, dtype = torch.float32)) if prior is not None else torch.zeros(len(CLASSES))
        self.register_buffer('log_prior', log_prior, persistent = False)   # .to(device)를 하면 같이 옮겨짐

    def forward(self, x) :
        probs = []
        for m in self.models :
            probs.append(m(x).softmax(1))
            if self.flip :
                probs.append(m(x.flip(3)).softmax(1))
        return torch.stack(probs).mean(0).clamp_min(1e-8).log() - self.log_prior


# ------------------------------------------------------------ 2. 가중치 불러오기

def load_model(weights_path, device) :
    # weights_path = 가중치 파일 1개의 경로, 또는 여러 파일의 목록 (예: ['final_model_1.pt', 'final_model_2.pt'])
    # 가중치는 CPU로 읽음 (.to(device)와 .eval()은 서버가 호출함)
    if isinstance(weights_path, (list, tuple)) :
        files = [torch.load(p, map_location = 'cpu', weights_only = True) for p in weights_path]
        for f in files :
            _check_saved(f)
        # 여러 파일의 모델을 하나로 합침 (좌우 뒤집기 여부와 클래스 비율은 모든 파일이 같음)
        saved = dict(files[0], states = [s for f in files for s in f['states']])
    else :
        saved = torch.load(weights_path, map_location = 'cpu', weights_only = True)

    # 이번 형식: 모델 여러 개의 가중치 목록('states') + 좌우 뒤집기 여부('flip') + 클래스 비율('prior', 없으면 보정 안 함)
    if isinstance(saved, dict) and 'states' in saved :
        _check_saved(saved)
        model = Ensemble(len(saved['states']), bool(saved.get('flip', False)), saved.get('prior'))
        for m, state in zip(model.models, saved['states']) :
            m.load_state_dict(state)            # fp16으로 저장한 가중치가 fp32 모델에 들어감
        return model

    # 이전 형식: 가중치 1개('state') 또는 state_dict만 저장한 파일
    if isinstance(saved, dict) and 'state' in saved :
        _check_saved(saved)
        state = saved['state']
    else :
        state = saved
    model = CNN3(n_classes = len(CLASSES))
    model.load_state_dict(state)
    return model


def _check_saved(saved) :
    # 가중치를 바꿨는데 상단 CLASSES / IMG_SIZE / MEAN / STD를 안 고치면
    # 오류 없이 점수만 떨어지므로, 값이 다르면 바로 오류를 냄
    problems = []
    if 'classes' in saved and list(saved['classes']) != CLASSES :
        problems.append(f"classes={saved['classes']}")
    if 'image_size' in saved and saved['image_size'] != IMG_SIZE :
        problems.append(f"image_size={saved['image_size']}")
    for key, ours in (('mean', MEAN), ('std', STD)) :
        if key in saved and any(abs(float(a) - b) > 1e-6 for a, b in zip(saved[key], ours)) :
            problems.append(f'{key}={list(saved[key])}')
    if problems :
        raise ValueError('가중치 파일과 .py 상단 값이 다름 (' + ', '.join(problems) + ') → 상단 값을 가중치 파일에 맞게 고쳐 주세요')


# ------------------------------------------------------------ 3. 전처리

# 휴대폰 사진의 회전 정보(EXIF)를 반영해 사진을 바로 세우고, RGB 3채널로 바꾸고,
# 비율과 관계없이 IMG_SIZE x IMG_SIZE 정사각형으로 맞춤 (BICUBIC)
class SquareResize :
    def __call__(self, img) :
        try :
            img = ImageOps.exif_transpose(img)
        except Exception :
            pass                                # EXIF 정보가 깨진 사진은 회전 없이 그대로 씀
        return img.convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)


def get_transform() :
    # 입력: PIL 이미지 1장 → 출력: Tensor [3, IMG_SIZE, IMG_SIZE]
    # ToTensor(0~1로 나눔) + Normalize = 노트북의 (x / 255 - MEAN) / STD
    return transforms.Compose([
        SquareResize(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# ------------------------------------------------------------ 제출 전 자체 점검 (서버가 불러올 때는 실행 안 됨)
# 사용법:  python model.py final_model_1.pt final_model_2.pt 아무_쓰레기_사진.jpg   → torch.Size([1, 12]), 예측 클래스, 확률 출력
#          (마지막 인자 = 사진, 그 앞은 모두 가중치 파일)

if __name__ == '__main__' :
    import sys
    net = load_model(sys.argv[1:-1], 'cpu').eval()
    x = get_transform()(Image.open(sys.argv[-1]).convert('RGB')).unsqueeze(0)
    with torch.no_grad() :
        out = net(x)
    print(out.shape, CLASSES[out.argmax(1).item()], f'{out.softmax(1).max().item():.3f}')
