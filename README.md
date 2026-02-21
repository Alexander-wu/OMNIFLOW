# OMNIFLOW L1 Reproduction

This repository provides an executable **L1 reproduction scaffold** for the paper:
`OMNIFLOW: A Physics-Grounded Multimodal Agent for Generalized Scientific Reasoning`.

## Scope

- End-to-end runnable pipeline
- Modular OMNIFLOW components:
  - `NES` (numerical simulator interface + baseline implementation)
  - `Visual Symbolic Projector`
  - `Physics Critic` (consistency constraints)
  - `Hierarchical Retrieval` (`Kphy`, `Kprot`, `Khist`)
  - `Agentic core` with a PG-CoT-like loop
- Reproducible evaluation for `RMSE`, `SSIM`, `PSNR`
- Structured scientific report generation

This is **not a strict bitwise reproduction** of the paper because key official assets (full code, weights, and complete configs) are not publicly available in the anonymous repository.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_l1_pipeline.py --config configs/l1_default.yaml
```

## Layered Config Workflow (Recommended for L2)

```bash
python scripts/validate_setup.py \
  --config configs/base.yaml \
  --config configs/data/era5.yaml \
  --config configs/model/nes_baseline.yaml \
  --config configs/exp/era5_l2.yaml
```

```bash
python scripts/run_l1_pipeline.py \
  --config configs/base.yaml \
  --config configs/data/era5.yaml \
  --config configs/model/nes_baseline.yaml \
  --config configs/exp/era5_l2.yaml \
  --dry-run
```

See `DATA_PREP.md` for expected dataset layout and tensor format.

## Outputs

- Predictions: `outputs/predictions.npy`
- Metrics: `outputs/metrics.json`
- Reports: `outputs/reports/`

## Project Layout

- `configs/`: experiment configuration
- `scripts/`: run/eval entry points
  - `validate_setup.py`: config and data-presence validation
  - `train_nes_placeholder.py`: training template (disabled by default)
  - `eval_multi_run.py`: aggregate multiple run metrics
- `src/omniflow_l1/`: core implementation
  - `data/`: dataset adapters and synthetic fallback
  - `nes/`: simulator interface and baseline model
  - `symbolic/`: visual-symbolic alignment module
  - `critic/`: physics consistency checks
  - `rag/`: retrieval pipeline
  - `agent/`: orchestration and report generation
  - `eval/`: metrics and evaluation
