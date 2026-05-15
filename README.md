# HDF5 Training Assets

이 폴더에는 학습 자료에서 무작위로 90,000개를 선택해 저장한 HDF5 파일과,
그 HDF5를 학습 데이터로 바로 사용할 수 있는 코드가 들어 있습니다.

## Files

https://drive.google.com/drive/folders/1uLP5QHQDX_8a1Y9X2FwqHIPGhnxEibRO?usp=drive_link

- `train_90000_random.h5`: 90,000개 랜덤 샘플을 저장한 HDF5
- `create_random_subset_hdf5.py`: HDF5 생성 스크립트
- `hdf5_training_dataset.py`: 정규화된 clean/error를 읽고 학습 시점에 noise를 추가하는 PyTorch Dataset/DataLoader 코드

## Source Used

기본값은 실제 학습에 사용된 캐시 데이터인 `/workspace/cached_data/train.pth` 입니다.
이 파일은 `/workspace/DATA/Train/*.out` 에서 생성된 학습용 캐시입니다.

원하면 raw `.out` 파일에서 직접 다시 뽑을 수도 있습니다.

## Download Prebuilt Artifacts

대용량 파일은 Google Drive에서 직접 받을 수 있습니다.

- `train_90000_random.h5`: https://drive.google.com/file/d/1nfqZpSXmpXErNUNK9X8jQSWTeRBh5CJi/view?usp=drive_link
- `final_model.ckpt`: https://drive.google.com/file/d/1-_TnN1Ba2QY7gLM6xHxxd5dfNvR4kORz/view?usp=drive_link

다운로드 후 기본 위치는 다음처럼 맞추면 됩니다.

- `train_90000_random.h5` -> repository root
- `final_model.ckpt` -> `figure_model_public/`

## HDF5 Format

- `wave`: `(N, L)`
- `clean_flux`: `(N, L)`
- `error`: `(N, L)`
- `sample_id`: `(N,)`
- `source_index`: `(N,)` when created from `.pth`

## Create Again

기본: cached train data에서 90,000개 추출

```bash
python /workspace/git/create_random_subset_hdf5.py \
  --input-pth /workspace/cached_data/train.pth \
  --output /workspace/git/train_90000_random.h5 \
  --n-samples 90000 \
  --seed 42
```

원본 `.out`에서 직접 생성

```bash
python /workspace/git/create_random_subset_hdf5.py \
  --glob "/workspace/DATA/Train/*.out" \
  --output /workspace/git/train_90000_random.h5 \
  --n-samples 90000 \
  --seed 42
```

## Use For Training

```python
from hdf5_training_dataset import create_dataloader

train_loader = create_dataloader(
    hdf5_path="/workspace/git/train_90000_random.h5",
    batch_size=64,
    min_sn=5,
    max_sn=20,
    shuffle=True,
    num_workers=4,
)
```

각 샘플은 다음 값을 반환합니다.

- `wave`
- `noisy_flux`
- `clean_flux`
- `error`
- `sn`

## Use With PyTorch Lightning

```python
from hdf5_lightning_datamodule import HDF5SpectrumDataModule

train_dm = HDF5SpectrumDataModule(
    hdf5_path="/workspace/git/train_90000_random.h5",
    batch_size=64,
    train_fraction=0.9,
    min_sn=5,
    max_sn=20,
    num_workers=4,
    seed=42,
)
```

기본 동작:
- 하나의 HDF5를 `train`/`val`로 랜덤 분할
- `train_fraction=0.9` 이면 81,000 / 9,000 분할
- 각 배치에서 `error / SN` 기준으로 noise를 동적으로 추가
- 반환 키: `wave`, `noisy_flux`, `clean_flux`, `error`, `sn`
