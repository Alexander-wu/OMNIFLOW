# Data Preparation Guide (L2)

This project expects each benchmark to provide one consolidated tensor file:

- `data/turbulence2d/dataset.npy`
- `data/era5/dataset.npy`
- `data/sevir/dataset.npy`

Tensor format must be:

- Shape: `[N, T, C, H, W]`
- Dtype: `float32`

## Expected Profiles

- `turbulence2d`: `H=128, W=128, C=2, T>=100`
- `era5`: `H=180, W=360, C=21, T>=200`
- `sevir`: `H=384, W=384, C=4, T>=50`

## Validation (No Training)

Use layered configs and run dry validation:

```bash
python scripts/validate_setup.py \
  --config configs/base.yaml \
  --config configs/data/era5.yaml \
  --config configs/model/nes_baseline.yaml \
  --config configs/exp/era5_l2.yaml
```

Or run pipeline dry-run:

```bash
python scripts/run_l1_pipeline.py \
  --config configs/base.yaml \
  --config configs/data/era5.yaml \
  --config configs/model/nes_baseline.yaml \
  --config configs/exp/era5_l2.yaml \
  --dry-run
```

## Notes

- When `use_synthetic_if_missing: false`, missing files will raise explicit errors.
- Keep all normalization/denormalization metadata outside `dataset.npy` in sidecar files if needed.
