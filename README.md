# BIESL Anomaly Detection

MIMIC-III-Ext-PPG의 30초 ECG/PPG 신호를 이용해 심장 리듬 이상을 두 단계로 분류하는 연구용 PyTorch 프로젝트다.

- Stage 1: 정상 동리듬(`SR`)과 그 밖의 이상 리듬을 이진 분류한다.
- Stage 2: 이상 리듬을 `atrial_tachy`, `conduction_block`, `paced`, `sinus_rate` 네 그룹으로 분류한다.
- 입력 modality: ECG, PPG 또는 ECG+PPG late fusion을 지원한다.
- 데이터 분할: fold 0–7 학습, fold 8 검증, fold 9 테스트를 사용한다.

> 이 저장소는 연구 프로토타입이다. 현재 평가·재현성상의 주의점과 모든 Python 코드의 줄별 해설은 [CODE_AUDIT_KO.md](CODE_AUDIT_KO.md)에 정리되어 있다. 특히 현재 Stage 2 평가는 Stage 1의 실제 예측이 아니라 정답 anomaly만 입력받으므로 진정한 end-to-end cascade 성능으로 해석하면 안 된다.

## 전체 처리 흐름

```text
metadata.csv + WFDB records
          │
          ▼
src/data/build_index.py ──► cache/index.parquet
          │
          ▼
src/data/dataset.py ──► train / validation / test Dataset
          │
          ▼
src/data/wfdb_signals.py ──► 30초 ECG/PPG tensor
          │
          ▼
src/models/factory.py
    ├─ Stage 1: ResNet1D
    ├─ Stage 2: CNNTransformer
    └─ both: LateFusionModel
          │
          ├─ src/train.py ──► checkpoints/*.pt
          └─ src/evaluate.py ──► 평가 지표와 confusion matrix
```

## 파일별 역할과 함수의 의미

각 절은 먼저 파일 전체의 책임을 설명하고, 그 안의 함수와 클래스가 파이프라인에서 어떤 의미인지 정리한다. 구문별 상세 동작은 각 절 마지막의 “줄별 해설” 링크에서 볼 수 있다.

### `main.py`

파일 역할: 프로젝트 생성 시 만들어진 최소 실행 예제다. 실제 데이터 처리·학습·평가에는 사용되지 않는다.

- `main()`: 고정된 인사 문구만 출력한다. 현재 연구 파이프라인의 진입점이 아니므로 향후 실제 CLI 안내로 교체하거나 제거할 수 있다.
- `if __name__ == "__main__"`: 이 파일을 직접 실행했을 때만 `main()`을 호출한다.

[이 파일의 줄별 해설](CODE_AUDIT_KO.md#mainpy)

### `src/config.py`

파일 역할: 데이터 위치, 결과 저장 위치, 표본 주파수, segment 길이, 채널명과 dataset fold를 한곳에서 정의한다. 다른 모듈이 동일한 실험 조건을 공유하게 하는 중앙 설정 파일이다.

함수나 클래스는 없다. `ROOT`, `DATA_ROOT` 등의 모듈 상수를 import해서 사용한다. `FS=125`와 `SEGMENT_LEN=3750`은 한 입력이 30초라는 뜻이며, `TRAIN_FOLDS`, `VAL_FOLDS`, `TEST_FOLDS`는 실험 분할 정책을 나타낸다.

[이 파일의 줄별 해설](CODE_AUDIT_KO.md#srcconfigpy)

### `src/utils.py`

파일 역할: 학습과 평가 양쪽에서 반복되는 device 이동과 class imbalance 보정 로직을 제공한다.

- `to_device(inputs, device)`: model 입력을 CPU에서 GPU 등 지정 device로 옮긴다. 단일 modality tensor와 fusion용 dictionary를 같은 방식으로 처리하기 위한 함수다. 반환값은 입력과 동일한 구조를 유지하면서 내부 tensor만 이동한 값이다.
- `compute_class_weights(labels, num_classes)`: 학습 레이블 빈도의 역수로 클래스 가중치를 만든다. 희귀 클래스의 loss 기여도를 높이기 위한 함수이며, `CrossEntropyLoss(weight=...)`에 바로 넣을 `float32` tensor를 반환한다.

[이 파일의 줄별 해설](CODE_AUDIT_KO.md#srcutilspy)

### `src/data/build_index.py`

파일 역할: 매우 큰 원본 `metadata.csv`를 chunk 단위로 읽어 로컬에 실제로 존재하는 환자만 남기고, 학습에 필요한 파생 레이블을 만든 뒤 `cache/index.parquet`으로 저장한다. 매 epoch마다 원본 CSV 전체를 다시 처리하지 않게 하는 전처리 단계다.

- `existing_patients()`: `dataset/mimic-data/p??/<patient>` 구조를 탐색해 로컬에 다운로드된 환자 ID 집합을 반환한다. metadata에는 있지만 신호 파일은 없는 환자를 사전에 거르기 위해 사용한다.
- `main()`: cache 디렉터리 생성, CSV chunk 로딩, 환자 필터링, `has_ecg`·`stage1_label`·`stage2_group` 생성, Parquet 저장과 분포 출력을 순서대로 수행하는 인덱스 빌드 진입점이다.

실행:

```bash
uv run python -m src.data.build_index
```

[이 파일의 줄별 해설](CODE_AUDIT_KO.md#srcdatabuild_indexpy)

### `src/data/labels.py`

파일 역할: 원본 `event_rhythm` 코드를 두 단계 cascade가 사용할 label로 변환하는 taxonomy를 정의한다. 어떤 원시 리듬을 같은 임상 그룹으로 볼 것인지 결정하므로 연구 결과의 의미에 직접 영향을 주는 파일이다.

- `stage2_group_for(event_rhythm)`: 원시 rhythm 문자열 하나를 Stage 2 그룹명으로 변환한다. `SR`, 미등록 코드와 현재 제외된 희귀 리듬은 `None`을 반환한다.
- `RHYTHM_GROUPS`: 원시 코드에서 네 그룹으로 가는 명시적 mapping이다.
- `STAGE2_CLASSES`: 그룹명을 정렬한 최종 class 목록이다.
- `STAGE2_CLASS_TO_IDX`: 문자열 class를 model 학습용 정수 index로 변환하는 mapping이다.

[이 파일의 줄별 해설](CODE_AUDIT_KO.md#srcdatalabelspy)

### `src/data/wfdb_signals.py`

파일 역할: WFDB record에서 지정 채널을 읽고, model이 받을 고정 길이 `float32` 배열로 바꾸며, segment별 정규화를 수행한다. metadata와 실제 waveform 사이의 경계 역할을 한다.

- `load_channels(folder_path, channels)`: 상대 record 경로와 채널명 목록을 받아 `{채널명: NumPy 배열}`을 반환한다. 요청 채널이 없으면 오류를 내며, 3,750 sample보다 긴 신호는 자르고 짧은 신호는 0으로 채운다.
- `normalize(x)`: 유한한 sample만으로 평균과 표준편차를 계산해 segment별 z-score 정규화를 수행한다. NaN/Inf 위치는 정규화 후 0으로 바꾸고, 전체가 유효하지 않으면 0 배열을 반환한다.

[이 파일의 줄별 해설](CODE_AUDIT_KO.md#srcdatawfdb_signalspy)

### `src/data/dataset.py`

파일 역할: Parquet 인덱스에서 split·modality·cascade stage 조건에 맞는 행을 고르고, 각 행을 PyTorch가 사용할 `(inputs, label)` 쌍으로 lazy-loading한다.

- `_load_full_index()`: `index.parquet`을 최초 한 번만 읽고 module cache에서 재사용한다. 인덱스가 없으면 생성 명령을 포함한 오류를 낸다.
- `_folds_for_split(split)`: `train`, `val`, `test`라는 이름을 설정 파일의 fold tuple로 변환한다.
- `_WFDBDataset`: 두 stage가 공유하는 내부 base Dataset이다.
  - `__init__()`: DataFrame, modality와 label 열을 저장하고 필요한 WFDB 채널을 결정한다.
  - `__len__()`: 현재 split의 표본 수를 반환한다.
  - `_build_inputs()`: record를 로드·정규화해 단일 tensor 또는 ECG/PPG dictionary로 만든다.
  - `__getitem__()`: 한 행의 waveform과 label을 반환한다. 현재는 로딩 실패 시 임의의 다른 행으로 재시도하므로 평가 재현성 측면에서 수정이 필요하다.
- `_maybe_limit()`: smoke test를 위해 고정 seed로 DataFrame을 지정 개수만큼 subsampling한다.
- `Stage1Dataset`: fold와 ECG 존재 여부를 적용하고 `stage1_label`을 반환하는 binary anomaly Dataset이다.
- `Stage2Dataset`: ground-truth anomaly이면서 네 그룹에 매핑된 행만 남기고 `stage2_idx`를 반환하는 rhythm 분류 Dataset이다.

[이 파일의 줄별 해설](CODE_AUDIT_KO.md#srcdatadatasetpy)

### `src/models/resnet1d.py`

파일 역할: 이미지용 ResNet의 residual 구조를 1차원 생체 신호에 맞게 바꾼 Stage 1 backbone을 정의한다. 모든 segment를 빠르게 screening하는 것이 목적이다.

- `BasicBlock1D`: 두 개의 1D convolution으로 feature를 변환한 뒤 원 입력을 더하는 residual block이다.
  - `__init__()`: main convolution 경로와 필요할 때 shape을 맞추는 downsample 경로를 만든다.
  - `forward(x)`: 두 convolution의 결과와 identity를 더해 gradient가 깊은 network를 안정적으로 통과하도록 한다.
- `ResNet1D`: stem, 네 residual stage, global average pooling과 classifier로 구성된 전체 network다.
  - `__init__()`: model 폭, stage별 block 수, dropout과 출력 class 수를 받아 layer를 조립한다.
  - `forward_features(x)`: 최종 classifier 직전의 고정 길이 embedding을 반환한다. fusion model이 이 method를 사용한다.
  - `forward(x)`: embedding에 dropout과 linear classifier를 적용해 class logits를 반환한다.

[이 파일의 줄별 해설](CODE_AUDIT_KO.md#srcmodelsresnet1dpy)

### `src/models/cnn_transformer.py`

파일 역할: convolution으로 국소 waveform 특징을 token화하고 Transformer self-attention으로 시간적인 rhythm 패턴을 통합하는 Stage 2 backbone을 정의한다.

- `ConvPatchEmbed`: raw 1D 신호를 더 짧은 embedding sequence로 바꾼다.
  - `__init__()`: stride 2 convolution 세 층을 조립한다.
  - `forward(x)`: `(batch, channel, time)`을 Transformer 형식 `(batch, token, embedding)`으로 변환한다.
- `PositionalEncoding`: attention 자체에는 없는 token 순서 정보를 고정 sine/cosine 값으로 추가한다.
  - `__init__()`: 최대 길이까지 positional table을 미리 계산해 buffer로 등록한다.
  - `forward(x)`: 실제 sequence 길이에 맞는 위치값을 입력 token에 더한다.
- `CNNTransformer`: convolution tokenization, CLS token, positional encoding, Transformer encoder와 classifier를 결합한 전체 Stage 2 model이다.
  - `__init__()`: embedding 폭, attention head 수, encoder layer 수와 classifier를 구성한다.
  - `forward_features(x)`: CLS token의 최종 표현을 rhythm embedding으로 반환한다.
  - `forward(x)`: embedding을 class logits로 변환한다.

[이 파일의 줄별 해설](CODE_AUDIT_KO.md#srcmodelscnn_transformerpy)

### `src/models/fusion.py`

파일 역할: ECG와 PPG를 서로 다른 encoder로 처리한 뒤 두 embedding을 합치는 late-fusion model을 정의한다. 두 센서의 형태 차이를 각 branch가 별도로 학습하게 하는 구조다.

- `LateFusionModel`: ECG encoder와 PPG encoder를 감싸는 multi-modal classifier다.
  - `__init__()`: 두 encoder를 등록하고 두 embedding의 합친 차원을 입력받는 MLP head를 만든다.
  - `forward(inputs)`: `inputs["ecg"]`와 `inputs["ppg"]`에서 각각 feature를 추출하고 연결한 뒤 최종 class logits를 반환한다.

[이 파일의 줄별 해설](CODE_AUDIT_KO.md#srcmodelsfusionpy)

### `src/models/factory.py`

파일 역할: stage와 modality라는 실험 설정만으로 올바른 model 구조를 만들어 주는 factory다. 학습·평가 코드가 구체적인 model class 조립법을 중복해서 알 필요가 없게 한다.

- `num_classes_for_stage(stage)`: Stage 1에는 2, Stage 2에는 현재 taxonomy의 class 수를 반환한다.
- `build_model(stage, modality, dropout=None)`: 단일 modality이면 해당 stage backbone 하나를 만들고, `both`이면 같은 종류의 ECG/PPG backbone 두 개와 `LateFusionModel`을 조립해 반환한다.
- `STAGE_BACKBONES`: stage 번호를 `ResNet1D` 또는 `CNNTransformer` class에 연결하는 registry다.

[이 파일의 줄별 해설](CODE_AUDIT_KO.md#srcmodelsfactorypy)

### `src/train.py`

파일 역할: CLI 인자로 실험 조건을 받아 Dataset/DataLoader/model/loss/optimizer/scheduler를 구성하고, validation macro-F1 기준으로 best checkpoint를 저장하는 학습 진입점이다.

- `build_datasets(stage, modality, limit)`: stage에 맞는 Dataset class를 골라 train과 validation Dataset을 함께 만든다.
- `evaluate(model, loader, device, autocast_dtype)`: gradient 없이 validation set의 예측을 모아 macro-F1 하나를 반환한다. 학습 중 best checkpoint 선택에 사용한다.
- `main()`: 인자 parsing부터 device 선택, loader/model/loss 구성, epoch·batch 학습, 중간/epoch-end 검증과 checkpoint 저장까지 전체 학습 수명주기를 실행한다.

예시:

```bash
uv run python -m src.train --stage 1 --modality ecg --epochs 10
uv run python -m src.train --stage 2 --modality both --epochs 10
```

[이 파일의 줄별 해설](CODE_AUDIT_KO.md#srctrainpy)

### `src/evaluate.py`

파일 역할: 저장된 best checkpoint를 test fold에 적용하고, Stage 1은 ROC-AUC·PR-AUC·classification report를, Stage 2는 다중 class report를 confusion matrix와 함께 출력한다.

- `collect_predictions(model, loader, device)`: gradient 없이 모든 batch의 logits를 softmax 확률로 바꾸고, 확률 행렬과 정답 배열을 반환한다.
- `main()`: CLI 조건으로 test Dataset과 model을 재구성하고 checkpoint weight를 불러온 뒤 stage에 맞는 metric을 계산·출력한다.

예시:

```bash
uv run python -m src.evaluate --stage 1 --modality ecg
uv run python -m src.evaluate --stage 2 --modality both
```

[이 파일의 줄별 해설](CODE_AUDIT_KO.md#srcevaluatepy)

### 빈 `__init__.py` 파일

파일 역할: `src`, `src.data`, `src.models` 디렉터리를 Python package로 명확하게 표시한다.

- `src/__init__.py`
- `src/data/__init__.py`
- `src/models/__init__.py`

세 파일 모두 함수, 클래스와 실행문이 없다.

[빈 package 파일의 줄별 해설](CODE_AUDIT_KO.md#빈-package-파일)

### `scripts/run_remaining_models.sh`

파일 역할: 아직 남은 stage/modality 조합을 순서대로 학습시키는 Bash 자동화 스크립트다. Python 파일은 아니지만 전체 실험 실행에 관여한다.

- `run(stage, modality)`: 두 인자를 받아 공통 hyperparameter로 `src.train`을 실행하고 시작·종료 상태를 출력한다.
- 아래 다섯 호출은 Stage 1 PPG/fusion과 Stage 2 ECG/PPG/fusion을 차례대로 학습한다.

## 설치와 데이터 준비

프로젝트는 Python 3.12 이상과 `uv` 사용을 전제로 한다.

```bash
uv sync
```

데이터는 기본적으로 다음 구조를 기대한다.

```text
dataset/mimic-data/
├─ metadata.csv
├─ p00/
│  └─ <patient directories>/
└─ ...
```

MIMIC 계열 데이터는 별도 접근 권한과 이용 조건이 적용될 수 있으므로 데이터 파일은 이 저장소에 포함하지 않는다.

## 결과와 코드 감사

- [ECG-only 성능 보고서](reports/ecg_only_performance.md)
- [ECG/PPG/fusion 비교 보고서](reports/ecg_ppg_fusion_comparison.md)
- [전체 Python 코드 줄별 해설 및 정당성 감사](CODE_AUDIT_KO.md)

논문이나 공식 결과에 사용하기 전에는 `CODE_AUDIT_KO.md`의 “매우 중요” 항목을 우선 해결해야 한다. 현재 저장소의 성능 보고서는 dataset과 checkpoint가 포함되어 있지 않아 이 checkout만으로 독립 재현된 수치가 아니다.
