# Python 코드 해설 및 논문용 코드 감사

검토 범위: 저장소에 추적된 Python 파일 16개(빈 `__init__.py` 3개 포함), 실행 스크립트와 보고서. 기준 커밋은 `852b29e`이다. 이 문서는 코드를 수정하지 않고 현재 상태를 설명하고 감사한 결과다.

## 1. 전체 실행 흐름

1. `src/data/build_index.py`가 원본 `metadata.csv`와 디스크의 환자 폴더를 대조해 `cache/index.parquet`을 만든다.
2. `src/data/dataset.py`가 fold 0–7/8/9를 train/validation/test로 나누고 Stage 1 또는 Stage 2 표본을 고른다.
3. `src/data/wfdb_signals.py`가 30초 ECG/PPG WFDB 레코드를 읽고 정규화한다.
4. `src/models/factory.py`가 Stage 1에는 1D ResNet, Stage 2에는 CNN+Transformer를 선택한다. `both`이면 ECG/PPG 두 인코더를 late fusion한다.
5. `src/train.py`가 class-weighted cross entropy로 학습하고 validation macro-F1이 가장 높은 체크포인트를 저장한다.
6. `src/evaluate.py`가 test fold에서 분류 보고서와 confusion matrix를 출력한다.

## 2. 파일별 줄 해설

빈 줄은 가독성만 위한 것이고 실행 효과가 없다. 아래에서 연속된 import, 상수 목록, 단순 출력문처럼 한 의미 단위인 줄은 범위로 묶었다.

### `main.py`

**파일 역할:** 프로젝트 생성 시 만들어진 최소 실행 예제다. 실제 데이터 전처리, model 학습과 평가에는 연결되지 않는다.

**함수·구문의 의미**

- `main()`: 인자를 받지 않고 고정된 인사 문구만 출력한다. 현재 연구 파이프라인의 진입 함수가 아니라 scaffold가 정상 실행되는지 보여 주는 placeholder다.
- `if __name__ == "__main__"`: 다른 파일에서 import할 때는 실행하지 않고 `python main.py`로 직접 실행했을 때만 `main()`을 호출하게 하는 Python 표준 진입 조건이다.

**줄별 코드 설명**

- 1: 패키지와 무관한 예제용 `main` 함수를 선언한다.
- 2: `Hello from biesl-anomaly!`만 출력한다. 실제 학습 파이프라인 진입점이 아니다.
- 5–6: 파일을 직접 실행할 때만 위 함수를 부른다.

### `src/config.py`

**파일 역할:** 데이터·cache·checkpoint 경로와 sampling 조건, 사용할 채널, train/validation/test fold를 중앙에서 정의한다. 다른 모듈이 동일한 실험 설정을 공유하도록 하는 구성 파일이다.

**함수·클래스의 의미:** 함수와 클래스는 없다. 다른 파일은 이 모듈을 import하고 `config.DATA_ROOT`, `config.SEGMENT_LEN`처럼 상수를 읽는다. 이 값이 바뀌면 데이터 위치, 입력 길이 또는 split 등 여러 단계의 동작이 함께 바뀐다.

**줄별 코드 설명**

- 1: 운영체제 독립 경로 처리를 위해 `Path`를 가져온다.
- 3: `src`의 부모, 즉 저장소 루트를 계산한다.
- 4–9: 데이터, metadata, cache/index, checkpoint, run 디렉터리 경로를 조합한다.
- 11: 모든 채널의 표본 주파수를 125 Hz로 가정한다.
- 12: 30초 길이를 3,750 sample로 고정한다.
- 13–14: 사용할 ECG와 PPG 채널명을 정한다.
- 16: 아래 fold가 환자 단위 분리라는 설명이다.
- 17–19: fold 0–7은 학습, 8은 검증, 9는 테스트로 고정한다.
- 21: 허용 센서 입력을 ECG, PPG, 둘 다로 제한한다.

### `src/utils.py`

**파일 역할:** 학습과 평가 양쪽에서 공통으로 필요한 device 이동과 class imbalance 보정 기능을 모아 둔다.

**함수의 의미**

- `to_device(inputs, device)`: model 입력을 CPU에서 GPU 등 지정 device로 옮긴다. 단일 tensor뿐 아니라 late fusion이 사용하는 `{"ecg": tensor, "ppg": tensor}` 구조도 동일하게 처리한다. 입력과 같은 자료구조를 반환하되 내부 tensor의 device만 달라진다.
- `compute_class_weights(labels, num_classes)`: class별 표본 수의 역수로 loss 가중치를 만들고 평균이 1이 되도록 맞춘다. 희귀 class의 오분류가 loss에 더 크게 반영되게 하며, `CrossEntropyLoss(weight=...)`에 넣을 `float32` tensor를 반환한다.

**줄별 코드 설명**

- 1–2: class weight 계산용 NumPy와 tensor/device 처리용 PyTorch를 가져온다.
- 5: 단일 tensor 또는 dictionary 입력을 device로 옮기는 함수를 선언한다.
- 6–7: 입력이 dictionary면 모든 값을 비동기 전송한다.
- 8: 아니면 입력 tensor 하나를 비동기 전송한다.
- 11: 레이블과 클래스 수로 loss weight를 만드는 함수를 선언한다.
- 12: 역빈도 가중치이며 평균을 1로 맞춘다는 설명이다.
- 13: 클래스별 개수를 세고 `float64`로 바꾼다.
- 14: 존재하지 않는 클래스의 0 division을 막기 위해 최소 개수를 1로 둔다.
- 15: 빈도의 역수를 구한다.
- 16: 전체 weight의 산술평균이 1이 되게 정규화한다.
- 17: PyTorch `float32` tensor로 반환한다.

### `src/data/build_index.py`

**파일 역할:** 큰 원본 `metadata.csv`와 로컬 WFDB 디렉터리를 대조해 실제 사용 가능한 행만 남기고, 두 단계 학습에 필요한 label을 붙여 `cache/index.parquet`으로 저장한다. 반복 학습 때마다 원본 CSV 전체를 다시 처리하지 않게 하는 사전 인덱싱 단계다.

**함수의 의미**

- `existing_patients() -> set[str]`: `DATA_ROOT/p??/<patient>` 디렉터리를 탐색한다. 입력 인자는 없으며, 현재 디스크에 존재하는 환자 디렉터리 이름들의 집합을 반환한다. metadata에는 존재하지만 신호가 다운로드되지 않은 환자를 제거하는 데 쓰인다.
- `main() -> None`: 인덱스 생성 작업의 진입 함수다. cache 디렉터리 생성 → 로컬 환자 조회 → CSV chunk 로딩 → 파생 열 생성 → Parquet 저장 → 분포 출력 순서로 전체 작업을 조정한다. 반환값은 없고 파일 생성과 console 출력이 부수 효과다.

**줄별 코드 설명**

- 1–6: 이 모듈의 목적, 파생 열, 실행 명령을 설명하는 docstring이다.
- 8–9: import 경로 조작을 위한 `sys`, `Path`를 가져온다.
- 11–12: 표 처리와 진행 막대를 가져온다.
- 14: 직접 모듈 실행 시 저장소 루트를 import 경로 앞에 삽입한다.
- 15–16: 전역 설정과 rhythm mapping을 가져온다.
- 18–26: 큰 CSV에서 실제로 읽을 7개 열을 정한다.
- 27: CSV를 한 번에 50만 행씩 읽는다.
- 30: 로컬에 존재하는 환자 ID 집합을 만드는 함수를 선언한다.
- 31: 빈 집합을 만든다.
- 32: `p??` 형태의 상위 폴더를 정렬해 순회한다.
- 33–36: 디렉터리만 골라 그 아래 환자 디렉터리 이름을 집합에 넣는다.
- 37: 환자 ID 집합을 반환한다.
- 40: 인덱스 생성 진입 함수다.
- 41: cache 디렉터리를 필요하면 만든다.
- 42–43: 디스크 환자 목록을 구하고 수를 출력한다.
- 45: CSV 전체 줄 수에서 header 한 줄을 빼 총 record 수를 센다.
- 46: 필터를 통과한 chunk를 모을 리스트다.
- 47: 전체 행 수 기준 진행 막대를 연다.
- 48: 지정 열만 chunk 단위로 읽는다.
- 49: 로컬에 데이터가 있는 환자의 행만 남긴다.
- 50: SQI 문자열에 정확히 `nan, nan, nan`이 포함되지 않으면 ECG가 있다고 간주한다.
- 51: rhythm이 `SR`이 아니면 Stage 1 anomaly(1)로 표시한다.
- 52: 원 rhythm 코드를 네 Stage 2 그룹 중 하나 또는 결측값으로 매핑한다.
- 53: 판정에 사용한 SQI 열을 제거한다.
- 54: 결과 chunk를 리스트에 보관한다.
- 55: 필터 전 행 수가 아니라 남은 행 수만큼 진행 막대를 증가시킨다.
- 57: 모든 chunk를 하나의 DataFrame으로 합친다.
- 58: Parquet 인덱스로 저장한다.
- 60–67: 저장 행 수, ECG 비율, 레이블과 fold 분포를 출력한다.
- 70–71: 모듈을 직접 실행할 때 `main`을 호출한다.

### `src/data/labels.py`

**파일 역할:** 원본 `event_rhythm` 코드를 Stage 1 정상/이상 label과 Stage 2의 네 rhythm 그룹으로 해석하기 위한 taxonomy를 정의한다. 어떤 원시 코드를 같은 class로 묶거나 제외할지를 결정하므로 연구 결과의 임상적 의미와 class 분포에 직접 영향을 준다.

**함수·상수의 의미**

- `stage2_group_for(event_rhythm) -> str | None`: 원시 rhythm 코드 하나를 받아 `sinus_rate`, `atrial_tachy`, `paced`, `conduction_block` 중 하나를 반환한다. `SR`, 제외한 희귀 rhythm 또는 알 수 없는 코드는 `None`을 반환한다.
- `RHYTHM_GROUPS`: 함수가 조회하는 원시 코드 → 그룹명 mapping이다.
- `STAGE2_CLASSES`: mapping에 실제로 존재하는 그룹명을 정렬한 최종 class 목록이다. 이 순서가 평가 보고서와 model 출력 index의 기준이 된다.
- `STAGE2_CLASS_TO_IDX`: 문자열 그룹명을 cross-entropy 학습에 필요한 0부터 시작하는 정수 index로 바꾼다.

**줄별 코드 설명**

- 1–14: 두 단계 레이블 정책과 희귀 클래스를 제외한 실험적 이유를 설명한다.
- 16: 정상 rhythm 원시 코드를 `SR`로 정의한다.
- 18: 다음 dictionary의 방향을 설명한다.
- 19–37: STACH/SBRAD/SARRH를 sinus rate, AF/AFLT/SVTACH/MATACH를 atrial tachy, VPACE/AVPACE/APACE를 paced, 1AVB/LBBB/RBBB를 conduction block으로 묶는다. 그 밖의 값은 매핑하지 않는다.
- 39: 네 그룹명을 중복 제거한 뒤 알파벳순으로 정렬한다.
- 40: 정렬 순서대로 0부터 class index를 부여한다.
- 43: 단일 원시 rhythm의 Stage 2 그룹을 찾는 함수를 선언한다.
- 44: 정상 또는 미등록 코드는 `None`이라는 설명이다.
- 45: dictionary 조회 결과를 반환한다.

### `src/data/wfdb_signals.py`

**파일 역할:** WFDB record를 실제 NumPy waveform으로 읽고, model 입력에 맞는 고정 길이와 자료형으로 보정한 뒤 segment별 정규화를 제공한다. 디스크의 의료 신호 형식과 PyTorch Dataset 사이의 경계다.

**함수의 의미**

- `load_channels(folder_path, channels) -> dict[str, np.ndarray]`: metadata의 상대 record 경로와 필요한 채널명 목록을 받는다. WFDB record에서 각 채널을 찾아 길이를 3,750 sample로 자르거나 0-padding하고 `{채널명: float32 배열}`로 반환한다. 요청 채널이 없으면 `ValueError`를 발생시킨다.
- `normalize(x) -> np.ndarray`: 한 segment의 유효 sample만으로 평균·표준편차를 계산해 z-score 정규화한다. NaN/Inf 위치는 정규화 평균인 0으로 치환하며, 결과는 원 shape의 `float32` 배열이다.

**줄별 코드 설명**

- 1–4: NumPy, WFDB와 전역 설정을 가져온다.
- 7: 한 record에서 요청 채널들을 읽는 함수를 선언한다.
- 8: 반환 형식과 고정 길이를 설명한다.
- 9: 데이터 루트와 metadata의 상대 경로를 합친다.
- 10: WFDB header/data record 전체를 읽는다.
- 11: record의 채널명 목록을 가져온다.
- 12: 물리 단위 신호 행렬을 가져온다.
- 14: 채널별 결과 dictionary를 만든다.
- 15: 요청 채널을 하나씩 처리한다.
- 16–17: 요청 채널이 없으면 상세한 예외를 낸다.
- 18: 해당 열을 골라 `float32`로 변환한다.
- 19: 길이가 3,750이 아닌지 검사한다.
- 20: 목표 길이의 0 배열을 만든다.
- 21: 원본과 목표 길이 중 짧은 쪽을 복사 길이로 정한다.
- 22: 앞부분을 복사해 긴 신호는 자르고 짧은 신호는 뒤를 0으로 채운다.
- 23: 보정 배열을 현재 채널로 사용한다.
- 24: 채널명 아래 결과를 넣는다.
- 25: 모든 채널 배열을 반환한다.
- 28: segment 단위 z-score 함수다.
- 29: NaN은 정규화 후 평균인 0으로 채운다는 설명이다.
- 30: 유한한 원소 mask를 만든다.
- 31–32: 전부 NaN/Inf면 0 배열을 반환한다.
- 33–34: 유한 원소의 평균과 표준편차를 계산한다.
- 35: 거의 상수인 신호는 0 division을 피하려 표준편차를 1로 바꾼다.
- 36: 전체 배열에서 평균을 빼고 표준편차로 나눈다.
- 37: 원래 유한하지 않았던 위치를 0으로 치환한다.
- 38: `float32`로 반환한다.

### `src/data/dataset.py`

**파일 역할:** Parquet 인덱스에서 요청한 split, modality와 cascade stage에 맞는 행을 고르고, 해당 WFDB 신호를 필요할 때 읽어 PyTorch `(inputs, label)` 표본으로 제공한다.

**함수·클래스의 의미**

- `_load_full_index() -> pd.DataFrame`: index Parquet을 최초 한 번만 읽고 module 전역 cache에 보관한다. 이후 Dataset 생성은 같은 DataFrame을 재사용한다.
- `_folds_for_split(split) -> tuple[int, ...]`: `train`, `val`, `test` 문자열을 설정 파일의 fold tuple로 바꾼다.
- `_WFDBDataset(Dataset)`: 두 stage Dataset이 공유하는 신호 로딩 base class다. 이름 앞의 `_`는 이 모듈 내부 구현용이라는 관례다.
  - `__init__(df, modality, label_col)`: 이미 필터된 DataFrame과 사용할 modality·정답 열을 저장하고 실제 채널 목록을 결정한다.
  - `__len__() -> int`: DataLoader가 사용할 전체 표본 수를 반환한다.
  - `_build_inputs(folder_path)`: WFDB 신호를 읽고 정규화한다. 단일 modality이면 `(1, 3750)` tensor, `both`이면 ECG/PPG tensor dictionary를 반환한다.
  - `__getitem__(idx)`: 한 index의 입력과 `long` label을 반환한다. 현재는 로딩 오류 시 다른 무작위 행으로 바꾸어 재시도한다.
- `_maybe_limit(df, limit, seed=0) -> pd.DataFrame`: 빠른 실험을 위해 DataFrame을 고정 seed로 최대 `limit`행까지 줄인다.
- `Stage1Dataset`: 지정 split에서 정상 `SR` 대 기타 rhythm의 이진 label을 제공한다. ECG가 필요한 경우 `has_ecg` 행만 사용한다.
- `Stage2Dataset`: ground-truth anomaly 중 네 taxonomy 그룹에 속한 행만 남기고 문자열 그룹을 정수 `stage2_idx`로 바꾼다.

**줄별 코드 설명**

- 1–9: 재시도 표본 추출, DataFrame, PyTorch Dataset, 설정·레이블·신호 함수를 가져온다.
- 11: 프로세스별 인덱스 DataFrame cache다.
- 14: 전체 인덱스를 lazy-load하는 함수를 선언한다.
- 15: module global을 수정하겠다고 알린다.
- 16: 아직 읽지 않았는지 검사한다.
- 17–20: parquet이 없으면 생성 명령이 포함된 오류를 낸다.
- 21: parquet 전체를 메모리에 읽는다.
- 22: cached DataFrame을 반환한다.
- 25: split 이름을 fold tuple로 바꾸는 함수다.
- 26: dictionary를 즉시 indexing하므로 잘못된 split은 `KeyError`가 난다.
- 29: 두 Stage가 공유할 private Dataset base class다.
- 30: subclass가 DataFrame과 label 열을 정한다는 설명이다.
- 32–34: type hint와 최대 로딩 시도 횟수다.
- 36: 이미 필터된 DataFrame, modality, label 열을 받는다.
- 37: modality가 허용값인지 assert한다.
- 38: 행 index를 0부터 다시 매긴다.
- 39–40: modality와 label 열을 저장한다.
- 41: modality를 실제 WFDB 채널 목록으로 변환한다.
- 43–44: Dataset 길이는 DataFrame 행 수다.
- 46: metadata 경로 하나를 model 입력으로 바꾸는 helper다.
- 47: 필요한 원시 채널을 읽는다.
- 48: 두 modality인지 검사한다.
- 49–52: ECG와 PPG를 각각 정규화해 `(1, 3750)` tensor dictionary로 반환한다.
- 53: 단일 modality의 실제 채널명을 고른다.
- 54: 정규화한 `(1, 3750)` tensor를 반환한다.
- 56: Dataset의 index 하나를 읽는 표준 method다.
- 57: 최대 다섯 번 시도한다.
- 58: 현재 index의 metadata 행을 읽는다.
- 59: 신호 로딩을 시도한다.
- 60: 입력 tensor를 만든다.
- 61: 같은 행의 label을 `long` tensor로 만든다.
- 62: 입력과 정답을 반환한다.
- 63: 종류를 가리지 않고 일반 `Exception`을 잡는다.
- 64–65: 마지막 시도도 실패하면 마지막 예외를 다시 던진다.
- 66: 그 전에는 원 요청과 무관한 무작위 index로 교체한다.
- 69: 데이터 수를 실험용으로 제한하는 helper다.
- 70–71: limit이 없거나 이미 작으면 그대로 반환한다.
- 72: 고정 seed로 정확히 limit개 행을 뽑는다.
- 75–76: Stage 1 dataset과 binary task 설명이다.
- 78: modality, split, 선택적 limit을 받는다.
- 79: 전체 index를 얻는다.
- 80: split fold만 남긴다.
- 81–82: ECG가 필요한 입력이면 heuristic `has_ecg`가 참인 행만 남긴다.
- 83: 요청되면 표본 수를 줄인다.
- 84: `stage1_label`을 정답으로 base class를 초기화한다.
- 87–88: anomaly rhythm 분류용 Stage 2 dataset이다.
- 90–92: 인자를 받고 해당 fold만 남긴다.
- 93: Stage 1의 실제 예측이 아니라 ground-truth anomaly만 남긴다.
- 94: 네 그룹에 매핑된 rhythm만 남긴다.
- 95–96: ECG가 필요하면 `has_ecg` 행만 남긴다.
- 97: 요청되면 수를 제한한다.
- 98: 새 열을 안전하게 쓰기 위해 복사한다.
- 99: 문자열 그룹을 정수 class index로 바꾼다.
- 100: `stage2_idx`를 정답으로 base class를 초기화한다.

### `src/models/resnet1d.py`

**파일 역할:** 이미지용 ResNet의 residual learning 구조를 1차원 ECG/PPG 신호에 맞게 구현한 Stage 1 backbone이다. 모든 30초 segment를 정상/이상으로 빠르게 screening한다.

**클래스·함수의 의미**

- `BasicBlock1D(nn.Module)`: 1D convolution 두 개로 특징을 변환한 뒤 원 입력을 더하는 residual block이다.
  - `__init__(in_ch, out_ch, stride=1)`: main convolution 경로를 만들고, 채널 수나 시간 길이가 바뀔 때 identity shape을 맞출 1×1 downsample 경로를 추가한다.
  - `forward(x)`: main 경로의 변환 결과와 identity를 더해 ReLU를 적용한다. 입력은 `(batch, channel, time)`이고 같은 batch와 호환되는 출력 feature map을 반환한다.
- `ResNet1D(nn.Module)`: stem, 네 residual stage, global average pooling, dropout과 classifier를 결합한 전체 분류 model이다.
  - `__init__(...)`: 입력 채널·class 수·기본 폭·stage별 block 수·dropout을 받아 network를 조립한다.
  - `forward_features(x)`: classifier 이전의 `(batch, embedding_dim)` feature를 반환한다. 단일 model의 분류와 fusion branch가 공유하는 표현이다.
  - `forward(x)`: feature에 dropout과 linear classifier를 적용해 `(batch, num_classes)` logits를 반환한다.

**줄별 코드 설명**

- 1: PyTorch neural-network layer API를 가져온다.
- 4: 1D residual basic block을 정의한다.
- 5–6: 입력/출력 채널과 stride를 받고 parent module을 초기화한다.
- 7–11: kernel 7 convolution 두 개, 각 BatchNorm, 공유 ReLU를 만든다.
- 13: 기본 residual 경로는 identity임을 표시한다.
- 14: 길이나 채널 수가 달라지는지 검사한다.
- 15–18: 다르면 kernel 1 convolution과 BatchNorm으로 identity의 shape을 맞춘다.
- 20: block 순전파를 정의한다.
- 21: 원 입력을 residual로 보존한다.
- 22: 첫 convolution → BatchNorm → ReLU를 계산한다.
- 23: 두 번째 convolution → BatchNorm을 계산한다.
- 24–25: 필요하면 residual도 downsample한다.
- 26: 두 경로를 더하고 ReLU를 적용한다.
- 29–30: ResNet-18 형태를 1D로 옮긴 classifier이며 Stage 1용이라는 설명이다.
- 32–39: 입력 채널, 클래스 수, 기본 폭, stage별 block 수, dropout 설정이다.
- 40: parent module을 초기화한다.
- 41–46: kernel 15/stride 2 convolution, BatchNorm, ReLU, max-pooling으로 stem을 만든다.
- 48: stage module을 모을 리스트다.
- 49: 현재 입력 채널 수를 기본 폭으로 시작한다.
- 50: stage별 block 수를 순회한다.
- 51: stage가 깊어질 때마다 출력 채널을 두 배로 한다.
- 52: 첫 stage 외에는 첫 block이 시간축 길이를 절반으로 줄인다.
- 53: 각 stage의 첫 block을 만든다.
- 54: 나머지는 shape을 유지하는 block으로 채운다.
- 55: block들을 한 stage로 묶는다.
- 56: 다음 stage 입력 채널을 갱신한다.
- 57: 모든 stage를 순차 module로 묶는다.
- 59: 시간축을 길이 1로 평균 pooling한다.
- 60: fusion이 사용할 feature 차원을 공개한다.
- 61–62: dropout과 최종 linear classifier를 만든다.
- 64: classifier 이전 feature 추출을 정의한다.
- 65–67: stem, residual stages, global pooling을 거쳐 `(batch, channels)`를 반환한다.
- 69–70: feature에 dropout과 classifier를 적용해 logits를 반환한다.

### `src/models/cnn_transformer.py`

**파일 역할:** convolution으로 국소 waveform 형태를 token sequence로 줄이고 Transformer self-attention으로 시간에 따른 rhythm 패턴을 통합하는 Stage 2 backbone이다.

**클래스·함수의 의미**

- `ConvPatchEmbed(nn.Module)`: 긴 raw 신호를 Transformer가 처리할 짧은 local feature token들로 변환한다.
  - `__init__(in_channels=1, d_model=128)`: stride 2 convolution 세 층을 만들어 시간 길이를 약 1/8로 줄이고 feature 폭을 `d_model`로 늘린다.
  - `forward(x)`: `(B, C, L)` waveform을 `(B, L', d_model)` token sequence로 반환한다.
- `PositionalEncoding(nn.Module)`: 순서 개념이 없는 attention에 각 token의 시간적 위치를 알려 주는 고정 sine/cosine encoding이다.
  - `__init__(d_model, max_len=2000)`: 최대 sequence 길이의 위치 table을 미리 계산하고 학습 대상이 아닌 buffer로 등록한다.
  - `forward(x)`: 입력 길이에 해당하는 위치값을 token에 더해 같은 shape으로 반환한다.
- `CNNTransformer(nn.Module)`: patch embedding, 학습 가능한 CLS token, positional encoding, Transformer encoder와 classifier를 합친 전체 rhythm classifier다.
  - `__init__(...)`: embedding 폭, head 수, encoder 층 수, feed-forward 폭, dropout과 class 수를 받아 model을 구성한다.
  - `forward_features(x)`: 전체 token을 encoder에 통과시키고 CLS 위치의 `(B, d_model)` 표현을 반환한다.
  - `forward(x)`: CLS 표현을 linear classifier에 넣어 `(B, num_classes)` logits를 반환한다.

**줄별 코드 설명**

- 1–4: positional encoding 수학, tensor, neural layer API를 가져온다.
- 7–8: raw waveform을 token sequence로 바꾸는 convolution front-end다.
- 10–11: 입력 채널과 embedding 폭을 받고 module을 초기화한다.
- 12–22: stride 2 convolution 세 층과 각 BatchNorm/ReLU를 만들어 길이를 약 1/8로 줄이고 채널을 `d_model`로 늘린다.
- 24–25: convolution 출력 `(B,C,L)`을 Transformer 입력 `(B,L,C)`로 바꾼다.
- 28: 고정 sinusoidal positional encoding module이다.
- 29–30: 폭과 최대 길이를 받고 초기화한다.
- 31: 위치 encoding 저장 배열을 만든다.
- 32: 위치 index 열벡터를 만든다.
- 33: 짝수 차원별 주파수 감소항을 계산한다.
- 34–35: 짝수 차원은 sine, 홀수 차원은 cosine으로 채운다.
- 36: batch 축을 추가하고 학습되지 않되 device와 함께 이동하는 buffer로 등록한다.
- 38–39: 현재 sequence 길이만큼 잘라 token에 더한다.
- 42–48: Stage 2 모델의 구성과 의도에 대한 설명이다.
- 50–59: 클래스 수, Transformer 폭/head/layer/feed-forward/dropout 설정이다.
- 60: parent module을 초기화한다.
- 61: convolution token extractor를 만든다.
- 62: 학습 가능한 한 개의 CLS token을 만든다.
- 63: positional encoding을 만든다.
- 64–71: batch-first, GELU를 쓰는 Transformer encoder layer를 만든다.
- 72: 해당 layer를 지정 횟수만큼 쌓는다.
- 73: CLS feature용 LayerNorm을 만든다.
- 74: fusion용 feature 차원을 공개한다.
- 75: 최종 class logits linear layer를 만든다.
- 77: classifier 전 feature 추출을 정의한다.
- 78: waveform을 token sequence로 바꾼다.
- 79: CLS token을 batch 수만큼 view 확장한다.
- 80: sequence 맨 앞에 CLS를 붙인다.
- 81: 위치 encoding을 더한다.
- 82: self-attention encoder를 통과시킨다.
- 83: 첫 token만 뽑아 정규화한 feature를 반환한다.
- 85–86: feature를 최종 classifier에 넣어 logits를 반환한다.

### `src/models/fusion.py`

**파일 역할:** ECG와 PPG를 별도 encoder로 처리하고 두 embedding을 합쳐 분류하는 late-fusion 구조를 정의한다. 센서별 waveform 특징은 따로 학습하고 최종 의미 표현에서 결합한다.

**클래스·함수의 의미**

- `LateFusionModel(nn.Module)`: ECG/PPG backbone 두 개와 fusion MLP head를 감싸는 multi-modal model이다.
  - `__init__(ecg_encoder, ppg_encoder, num_classes, hidden_dim=128, dropout=0.2)`: 두 encoder를 submodule로 등록하고, 두 `embedding_dim`의 합을 입력받는 hidden linear layer와 최종 classifier를 만든다.
  - `forward(inputs)`: `inputs["ecg"]`와 `inputs["ppg"]`를 각각 `forward_features`에 넣고, 두 feature를 연결해 `(batch, num_classes)` logits를 반환한다.

**줄별 코드 설명**

- 1–2: tensor 연결과 neural layer API를 가져온다.
- 5–11: ECG/PPG 별도 인코더 feature를 합치는 late-fusion 의도를 설명한다.
- 13: 두 encoder, 클래스 수, head 폭/dropout을 받는다.
- 14–16: parent 초기화 후 encoder를 submodule로 등록한다.
- 17: 두 feature 차원의 합을 계산한다.
- 18–23: concat feature를 hidden layer, ReLU, dropout, 출력 layer에 통과시키는 head다.
- 25: dictionary 입력의 순전파를 정의한다.
- 26–27: 각 센서 tensor를 해당 encoder의 classifier 이전 feature로 바꾼다.
- 28: 두 feature를 channel 차원으로 연결한다.
- 29: fusion head가 class logits를 만든다.

### `src/models/factory.py`

**파일 역할:** stage와 modality 설정만 받아 올바른 backbone과 fusion wrapper를 만들어 주는 model factory다. 학습·평가 코드가 구체적인 class 조립 방식을 중복하지 않게 한다.

**함수·상수의 의미**

- `STAGE_BACKBONES`: Stage 1을 `ResNet1D`, Stage 2를 `CNNTransformer` class에 연결하는 registry다.
- `num_classes_for_stage(stage) -> int`: Stage 1이면 2, Stage 2이면 현재 taxonomy의 그룹 수를 반환해 model 출력 차원을 정한다.
- `build_model(stage, modality, dropout=None) -> nn.Module`: 단일 modality이면 1-channel backbone 하나를 반환한다. `both`이면 ECG/PPG backbone을 각각 만들고 `LateFusionModel`로 감싸 반환한다.

**줄별 코드 설명**

- 1–6: module type, Stage 2 클래스 목록과 세 model 구현을 가져온다.
- 8–11: 단계별 backbone 선택 이유를 서술한다.
- 12: Stage 1을 ResNet1D, Stage 2를 CNNTransformer에 연결한다.
- 15: 단계별 클래스 수 helper다.
- 16: Stage 1이면 2, 그 밖이면 현재 Stage 2 그룹 수를 반환한다.
- 19: 단계와 modality에 맞는 model 생성 함수다.
- 20–21: 허용 stage와 modality를 assert한다.
- 22: 단계에 맞는 class object를 고른다.
- 23: 출력 클래스 수를 구한다.
- 24: dropout이 명시됐을 때만 constructor keyword로 전달한다.
- 26–27: 단일 센서이면 1-channel backbone 하나를 반환한다.
- 29–30: 두 센서이면 동일 종류의 ECG/PPG backbone을 각각 만든다.
- 31: 두 backbone을 late-fusion wrapper에 넣는다.

### `src/train.py`

**파일 역할:** command-line 인자로 stage와 modality·hyperparameter를 받아 Dataset, DataLoader, model, loss, optimizer와 scheduler를 구성하고 학습한다. validation macro-F1이 가장 높은 model과 마지막 model을 checkpoint로 저장하는 학습 진입점이다.

**함수의 의미**

- `build_datasets(stage, modality, limit)`: stage에 맞는 Dataset class를 선택해 train/validation Dataset 두 개를 반환한다. smoke-test limit이 있으면 train과 더 작은 validation subset에 적용한다.
- `evaluate(model, loader, device, autocast_dtype)`: 학습 중 validation 전용 평가 함수다. gradient를 끄고 전체 loader의 class 예측과 정답을 모아 macro-F1 scalar를 반환한다. best checkpoint 선택 기준으로 사용한다.
- `main()`: CLI parsing → device 선택 → loader/model/loss/optimizer/scheduler 생성 → batch 학습 → 선택적 중간 검증 → epoch 검증 → checkpoint 저장까지 전체 training loop를 실행한다. 반환값 대신 checkpoint 파일과 console log를 만든다.

**줄별 코드 설명**

- 1–11: CLI/time, PyTorch, macro-F1, DataLoader와 프로젝트 구성요소를 가져온다.
- 14: 단계에 맞는 train/validation dataset을 만드는 helper다.
- 15: stage가 1이면 Stage1Dataset, 아니면 Stage2Dataset을 고른다.
- 16: train split에는 요청 limit을 적용한다.
- 17: validation은 full이거나 train limit의 1/5(최소 100)를 쓴다.
- 18: 두 dataset을 반환한다.
- 21: 아래 평가에서 gradient 기록을 끈다.
- 22: validation macro-F1 함수다.
- 23: dropout/BatchNorm을 평가 모드로 둔다.
- 24: 예측과 정답 batch를 모을 리스트다.
- 25: validation loader를 순회한다.
- 26: 입력을 device로 옮긴다.
- 27: CUDA일 때 bfloat16 autocast context를 연다.
- 28: logits를 계산한다.
- 29: 최대 logit class를 CPU에 모은다.
- 30: CPU에 있던 정답을 모은다.
- 31–32: batch들을 합쳐 NumPy 배열로 바꾼다.
- 33: 0 division을 0으로 처리한 macro-F1을 반환한다.
- 36: CLI 진입 함수다.
- 37: argument parser를 만든다.
- 38–48: stage, modality, epoch, batch, learning rate, weight decay, dropout, warmup 비율, 중간 검증 간격, worker 수, smoke-test limit 인자를 정의한다.
- 49: CLI를 parsing한다.
- 51: CUDA가 있으면 GPU, 아니면 CPU를 고른다.
- 52: mixed precision 자료형을 bfloat16으로 고정한다.
- 54: train/validation dataset을 만든다.
- 55: 크기와 실험 조건을 출력한다.
- 57–60: train loader를 shuffle, pinned memory, drop-last와 선택적 persistent worker로 만든다.
- 61–64: validation loader는 shuffle/drop-last 없이 만든다.
- 66: model을 만들고 device로 옮긴다.
- 67: 출력 클래스 수를 구한다.
- 69: 단계별 정답 열 이름을 고른다.
- 70: train label 빈도로 loss class weight를 계산해 device로 옮긴다.
- 71: weighted cross entropy loss를 만든다.
- 72: AdamW optimizer를 만든다.
- 73–75: 전체 예상 step 수와 warmup 비율로 OneCycle learning-rate scheduler를 만든다.
- 77: checkpoint 디렉터리를 만든다.
- 78: best checkpoint 파일명을 정한다.
- 79: 아직 best가 없으므로 F1을 -1로 시작한다.
- 80: epoch을 넘는 step counter다.
- 82: epoch을 1부터 지정 횟수까지 순회한다.
- 83: model을 학습 모드로 둔다.
- 84: epoch 시간을 재기 시작한다.
- 85: 누적 loss를 0으로 시작한다.
- 86: train batch를 1부터 세며 순회한다.
- 87: global step을 증가시킨다.
- 88–89: 입력과 정답을 device로 옮긴다.
- 91: 이전 gradient를 `None`으로 초기화한다.
- 92–94: 선택적 mixed precision으로 logits와 loss를 계산한다.
- 95: 역전파로 gradient를 계산한다. AMP scaler는 사용하지 않는다.
- 96: parameter를 갱신한다.
- 97: learning rate를 한 step 진행한다.
- 99: 출력용 누적 loss에 현재 값을 더한다.
- 100–101: 50 step마다 epoch 평균 loss와 현재 learning rate를 출력한다.
- 103: 중간 검증 설정이 있고 해당 global step인지 검사한다.
- 104: validation macro-F1을 계산한다.
- 105: 평가가 바꾼 model mode를 다시 train으로 복구한다.
- 106: 중간 성능을 출력한다.
- 107–110: 지금까지 최고면 best F1을 갱신하고 model 및 CLI 인자를 저장한다.
- 112: epoch 끝 validation F1을 계산한다.
- 113: 걸린 시간을 계산한다.
- 114: epoch 요약을 출력한다.
- 116: 성능과 무관하게 latest checkpoint를 저장한다.
- 117–120: epoch 결과가 최고면 best checkpoint도 갱신한다.
- 123–124: 직접 실행할 때 `main`을 호출한다.

### `src/evaluate.py`

**파일 역할:** 저장한 best checkpoint를 test fold에 적용해 최종 성능 지표와 confusion matrix를 출력한다. 학습 중 model 선택용 validation 함수와 달리 논문 결과 산출을 위한 test 평가 진입점이다.

**함수의 의미**

- `collect_predictions(model, loader, device)`: gradient 없이 모든 test batch를 순회한다. logits를 softmax class 확률로 바꾸어 `probs` 행렬과 `labels` 배열을 반환한다.
- `main()`: CLI 조건으로 test Dataset과 model 구조를 만들고 checkpoint weight를 로드한다. Stage 1이면 ROC-AUC·PR-AUC·이진 보고서를, Stage 2이면 네 class classification report를 confusion matrix와 함께 출력한다.

**줄별 코드 설명**

- 1–12: CLI, PyTorch softmax, sklearn metric, DataLoader와 프로젝트 구성요소를 가져온다.
- 15: prediction 수집 중 gradient 기록을 끈다.
- 16: model과 loader로 확률·정답을 모으는 함수다.
- 17: model을 평가 모드로 둔다.
- 18: batch 결과 리스트를 만든다.
- 19: 평가 loader를 순회한다.
- 20: 입력을 device로 옮긴다.
- 21–22: CUDA에서 bfloat16으로 logits를 계산한다.
- 23: logits를 float32 확률로 바꿔 CPU로 옮긴다.
- 24–25: 확률과 정답을 모은다.
- 26: batch들을 합쳐 NumPy 배열로 반환한다.
- 29: CLI 진입 함수다.
- 30: argument parser를 만든다.
- 31–36: stage, modality, batch, worker, limit, checkpoint 경로 인자를 정의한다.
- 37: CLI를 parsing한다.
- 39: 실행 device를 고른다.
- 40: 단계에 맞는 Dataset class를 고른다.
- 41: test split dataset을 만든다.
- 42: 순서를 섞지 않는 loader를 만든다.
- 44: 지정 경로 또는 기본 best checkpoint 경로를 정한다.
- 45: checkpoint를 device에 맞춰 읽는다.
- 46: 현재 코드의 기본 설정으로 model을 다시 만든다.
- 47: 저장 parameter를 model에 로드한다.
- 49: 확률과 정답을 계산한다.
- 50: 최고 확률 class를 예측으로 고른다.
- 52: stage/modality/평가 수를 출력한다.
- 53: Stage 1인지 검사한다.
- 54: anomaly class 확률만 고른다.
- 55: 두 class 표시 이름을 정한다.
- 56–61: ROC-AUC, PR-AUC, class별 보고서와 2×2 confusion matrix를 출력한다.
- 62: Stage 2 분기다.
- 63: 현재 Stage 2 class index 전체를 만든다.
- 64–67: 네 class 분류 보고서와 confusion matrix를 출력한다.
- 70–71: 직접 실행할 때 `main`을 호출한다.

### 빈 package 파일

**파일 역할:** `src`, `src.data`, `src.models` 디렉터리를 Python package로 명확하게 표시한다. 세 파일은 모두 비어 있다.

**함수·클래스의 의미:** 정의된 함수나 클래스가 없으며 runtime 동작도 없다. 패키지 인식과 향후 package-level export 위치로만 존재한다.

**줄별 코드 설명:** 0줄이므로 설명할 실행문이 없다.

- `src/__init__.py`, `src/data/__init__.py`, `src/models/__init__.py`: 모두 0줄이다. 디렉터리를 Python package로 명확히 표시할 뿐 실행 로직은 없다.

## 3. 정당성·재현성 감사 결과

### 매우 중요

1. **현재 평가는 진짜 end-to-end cascade 평가가 아니다.** `Stage2Dataset` 93행은 Stage 1이 실제로 검출한 표본이 아니라 정답이 anomaly인 모든 표본을 Stage 2에 준다. 따라서 Stage 1 false negative와 false positive가 Stage 2 최종 결과에 반영되지 않는다. 그런데 보고서는 “evaluated end to end”라고 표현한다. 논문에는 단계별 conditional 성능이라고 써야 하며, 실제 cascade 전체 confusion/coverage도 별도로 계산해야 한다.
2. **test fold가 사실상 모델 개발에 사용된 흔적이 있다.** 보고서는 여러 시도 간 test ROC-AUC를 비교해 “deployed checkpoint”와 클래스 정책을 정했다고 서술한다. 특히 희귀 클래스를 “모든 modality에서 0% F1”이라며 제거한 결정이 test 결과를 본 뒤 이루어졌다면 test leakage다. taxonomy와 hyperparameter 결정은 train/validation만으로 확정하고 test는 마지막 한 번만 사용해야 한다.
3. **로드 실패를 무작위 다른 표본으로 조용히 교체한다.** `dataset.py` 56–66행 때문에 평가 중에도 실패 표본이 중복된 임의 표본으로 바뀐다. 측정 대상과 표본 수가 불투명해지고 재현되지 않는다. 실패 목록을 사전에 검증·제외하고 고정 manifest를 남기거나, 평가에서는 즉시 실패시켜야 한다.
4. **환자 단위 독립성은 코드가 검증하지 않는다.** `strat_fold`를 신뢰할 뿐 `subject_id`가 split 사이 중복되지 않는지 assert/report가 없다. 논문 핵심 주장이라면 build 단계에서 교집합이 0인지 자동 검증해야 한다.

### 중요

5. **Stage 1 label의 결측 처리 위험:** `event_rhythm != "SR"`는 결측값도 참으로 만들어 anomaly로 분류할 수 있다. 결측/unknown 정책을 명시하고 별도 제외 또는 라벨링해야 한다.
6. **`has_ecg`가 실제 ECG 존재 여부를 보장하지 않는다.** 특정 SQI 문자열 패턴 하나만 검사한다. 실제 `II` 채널 존재, 유효 sample 비율, SQI threshold를 검증하지 않아 runtime 교체 로직에 의존하게 된다.
7. **짧은 record padding이 정규화를 왜곡한다.** 0 padding을 먼저 한 뒤 모든 0을 정상 관측값으로 보고 평균/표준편차를 계산한다. 원 길이 구간만 정규화한 뒤 padding하거나 valid mask를 사용해야 한다. Transformer에도 padding mask가 없다.
8. **재현성 설정이 없다.** Python/NumPy/PyTorch seed, worker seed, deterministic 설정, 실행별 config snapshot, dataset/index hash가 없다. `_maybe_limit`만 seed가 고정되어 있다.
9. **독립 표본 가정이 강하다.** 수십만 30초 segment가 같은 환자에서 반복될 수 있는데 confidence interval이나 patient-level bootstrap이 없다. 단순 segment 수를 근거로 성능 확실성을 과장할 수 있다.
10. **Stage 2 제외 표본의 임상적 처리와 coverage가 불명확하다.** JR/JTACH/VTACH와 모든 미등록 코드는 Stage 1 anomaly지만 Stage 2는 답을 내지 않는 것으로 데이터셋에서는 제외된다. 실제 inference에는 reject/unknown class나 abstention 경로가 구현되어 있지 않다.
11. **checkpoint provenance가 부족하다.** model weight와 CLI 인자만 저장하고 code commit, label 순서, optimizer/scheduler, epoch/step, best metric, package version, random seed를 저장하지 않는다. 현재 taxonomy가 바뀌면 옛 checkpoint를 안전하게 해석하기 어렵다.

### 보통/정리 권장

12. `build_index.py` 55행의 progress는 raw chunk 크기가 아니라 필터 후 크기만 더해 100%에 도달하지 않을 수 있다.
13. `pd.concat(chunks)`는 읽을 chunk/대상 환자가 하나도 없을 때 명확한 진단 대신 실패한다. CSV 파일도 context manager와 명시적 encoding으로 여는 편이 낫다.
14. `factory.py`의 `num_classes_for_stage`는 1이 아닌 잘못된 stage를 모두 Stage 2로 취급한다. 호출부 assert에 기대지 말고 함수 자체가 검증해야 한다.
15. fusion의 두 backbone에는 호출되지 않는 classifier layer가 그대로 들어 있다. gradient는 생기지 않지만 parameter/state_dict가 불필요하게 커지고 구조가 덜 명료하다.
16. fusion docstring은 single-modality weight로 warm-start할 수 있다고 말하지만 학습 코드에는 warm-start 기능이 없다. 구현 전에는 “향후 가능”으로 표현해야 한다.
17. CNNTransformer constructor 기본 `num_classes=6`은 현재 네 클래스와 어긋난 낡은 기본값이다. factory에서는 덮어써 동작하지만 직접 생성 시 잘못된다.
18. 모든 입력 오류 검증이 `assert`라 최적화 실행(`python -O`)에서 사라진다. 사용자 입력에는 `ValueError`가 적합하다.
19. CUDA bfloat16 지원 여부를 검사하지 않는다. 일부 GPU에서는 실패하거나 예상보다 느릴 수 있다.
20. README가 비어 있고 `main.py`와 `pyproject.toml` description은 scaffold placeholder다. 반면 실제 실행은 `python -m src...`이다. 이 조합은 “급하게 생성한 프로젝트” 인상을 강하게 준다.
21. `RUNS_DIR`와 tensorboard/ONNX 관련 dependency는 현재 코드에서 쓰지 않는다. 실제 필요가 없다면 제거하거나 사용 목적을 문서화해야 한다.
22. 자동 테스트와 CI가 없다. 최소한 label mapping, normalization, 길이 보정, split disjointness, model output shape, checkpoint round-trip 테스트가 필요하다.
23. 보고서에 문자 인코딩 깨짐(`??`, `쨌`, `짹`)이 많고, 한 보고서는 과거 6-class 결과를 현재 설계 설명처럼 제시한다. 논문 자료로 쓰기 전에 UTF-8 복구와 결과 버전 표기가 필요하다.

## 4. 복사·AI 생성 흔적 판단

- 공개 웹에서 `BasicBlock1D`, `ConvPatchEmbed`, late-fusion 설명, `stage2_group_for`의 고유 문구 조합을 검색했으나 동일 문구/파일은 찾지 못했다. 따라서 **노골적인 인터넷 코드 통째 복사 증거는 발견되지 않았다.** 다만 웹 검색은 private repository, 논문 부록, 변경된 변수명의 원본까지 포괄하지 못하므로 비표절의 증명은 아니다.
- residual block은 He et al.의 ResNet, sine/cosine positional encoding과 Transformer encoder는 Vaswani et al.의 Transformer에서 온 표준 아이디어다. PyTorch API 형태도 공식 예제와 자연히 비슷할 수 있다. 논문과 저장소 README에서 두 원 논문 및 PyTorch 구현 기반임을 인용하면 문제될 성격이 아니다.
- “바이브 코딩”처럼 보일 가능성이 높은 부분은 알고리즘 자체보다 **프로젝트 마감 상태**다: 빈 README, Hello-world main, placeholder description, 사용하지 않는 dependency/상수, 구현되지 않은 warm-start를 설명하는 주석, 낡은 6-class 기본값, 깨진 보고서 문자, 테스트 부재가 서로 겹친다.
- 주석 중 “ECG와 PPG가 달라 shared early convolution이 capacity를 낭비한다”, “beat-to-beat attention이 특정 군을 구분한다”, “clinically coherent buckets” 같은 문장은 가능한 가설이지 이 코드가 입증한 사실이 아니다. 논문에서는 인용 또는 ablation 근거를 붙이고, 없으면 단정 대신 설계 가설로 낮춰 써야 한다.
- 커밋 이력은 세 커밋 모두 동일 저자이며 초기 파이프라인, 보고서, 클래스 변경 순이다. 이력만으로 외부 출처나 작성 방식은 판별할 수 없다.

## 5. 논문 제출 전 권장 우선순위

1. taxonomy와 모든 선택을 validation에서 다시 확정하고 untouched test set으로 최종 1회 평가한다.
2. 실제 Stage 1 예측을 Stage 2에 전달하는 end-to-end evaluator와 reject/unknown 정책을 구현한다.
3. 환자 split 검증, 실패 record manifest, seed/config/data hash를 자동 기록한다.
4. README에 데이터 출처·라이선스·전처리·모델·명령·인용·결과 재현법을 쓴다.
5. 위 최소 테스트를 추가하고 불필요 scaffold/dependency 및 과장된 주석을 정리한다.
6. patient-level bootstrap confidence interval과 class/환자별 support를 함께 보고한다.

## 6. 이번 검증에서 실제로 확인한 것과 한계

- 모든 Python 파일은 Python 3.13.7의 `compileall` 문법 검사를 통과했다.
- 현재 환경에 `wfdb`가 설치되어 있지 않고 `uv` 실행 파일도 없어 dependency를 포함한 model forward/data smoke test는 수행하지 못했다.
- dataset과 checkpoint가 저장소에 포함되어 있지 않아 보고서 수치를 재계산하지 못했다. 따라서 보고서의 성능 숫자는 코드만으로 독립 검증된 값이 아니다.
- 유사성 판단은 공개 웹의 고유 문구 검색과 코드 구조 검토 수준이다. 엄밀한 provenance가 필요하면 작성자에게 원본 notebook/실험 로그/참고 repository 목록을 받아 commit별 diff와 대조해야 한다.
