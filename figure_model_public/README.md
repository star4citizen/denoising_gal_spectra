# Published Figure Model

이 폴더는 `Figure` 안의 그림 생성과 분석에 사용한 최종 학습 모델을 공개하기 위한 최소 구성입니다.

## Files

- `published_model.py`: 공개용 추론 전용 모델 정의와 로더
- `load_public_model.py`: 체크포인트 로드와 간단한 추론 예제
- `README.md`: 공개용 사용법

체크포인트 파일은 이 폴더 안의 `final_model.ckpt` 를 기본으로 사용합니다.

## Model Used In Figure Scripts

`Figure` 폴더의 분석 스크립트들은 모두 아래 체크포인트를 사용합니다.

- `/workspace/git/figure_model_public/final_model.ckpt`

이 파일은 Figure 분석에서 사용된 실제 최종 체크포인트이며,
루트의 `checkpoints_fourier_edge/model-fourier_freq1874-epoch=974-val_loss=0.0385.ckpt`를 복사한 동일한 모델입니다.

## Quick Start

```python
from published_model import load_model, denoise_flux

model = load_model()
```

단일 스펙트럼 추론:

```python
import numpy as np
from published_model import load_model, denoise_flux

model = load_model()
noisy_flux = np.random.randn(3748).astype(np.float32)
denoised_flux = denoise_flux(model, noisy_flux)
```

## Example Script

```bash
python /workspace/git/figure_model_public/load_public_model.py
```

## Checkpoint Summary

기본 하이퍼파라미터:
- `enc_ch1=96`
- `enc_ch2=192`
- `d_model=384`
- `nhead=12`
- `num_transformer_layers=8`
- `dim_feedforward=1536`
- `dec_ch2=192`
- `dec_ch1=96`
- `core_output_channels=64`
- `dropout=0.15`
- `fourier_lambda=0.2`
- `num_low_freqs=1874`
