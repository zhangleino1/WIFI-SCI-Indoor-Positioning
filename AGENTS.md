# Repository Guidelines

## Project Structure & Module Organization
This repository is a flat Python project centered on model training in `main.py`. Core data loading lives in `csi_dataset.py`; model definitions are split across `cnn_net_model.py`, `cnn_lstm_net_model.py`, and `cnn_transformer_model.py`. Analysis and visualization scripts such as `visualize_classification.py`, `visualize_locations.py`, and `analyze_spatial_confusion.py` generate plots into `results/` or `logs/`. Sample CSVs for quick checks live in `dataset/`; the bundled set is partial, not the full dataset. Reference docs are `readme.md`, `USAGE_GUIDE.md`, and `csi.pdf`.

## Build, Test, and Development Commands
Use Python 3.8+ with PyTorch and Lightning.

```powershell
conda create -n csi-positioning python=3.8
conda activate csi-positioning
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install lightning -c conda-forge
pip install scikit-learn matplotlib seaborn pandas tensorboard tensorboardX
python main.py --model_type cnn --data_dir ./dataset --mode train
python main.py --model_type cnn_lstm --data_dir ./dataset --mode test --cpt_path ./logs/cnn_lstm/version_0/checkpoints/last.ckpt
python visualize_classification.py --model_path ./logs/cnn/version_0/checkpoints/last.ckpt --model_type cnn --data_dir ./dataset
python simple_test.py
```

`visualize_classification.py` keeps its old filename for compatibility, but it now evaluates regression checkpoints and writes distance-error plots rather than confusion matrices.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, module-level imports, and small top-level scripts with `argparse`. Use `snake_case` for files, functions, variables, and CLI flags; use `PascalCase` for dataset and model classes such as `CSIDataset` and `CNN_LSTM_Net`. Keep comments brief and practical. No formatter or linter config is committed, so match the surrounding code and keep new dependencies explicit in docs.

## Testing Guidelines
There is no full automated test suite yet. Treat `simple_test.py` as a smoke test for checkpoint loading and `(batch, 2)` regression outputs, and run at least one train or test command against `./dataset` before opening a PR. When changing preprocessing, model outputs, or evaluation code, include the exact command used and summarize the metric or artifact produced.

## Commit & Pull Request Guidelines
Recent history mixes plain subjects (`Update readme.md`) with conventional prefixes (`feat:`, `fix:`, `docs:`, `chore:`). Prefer short, imperative commit messages, ideally `type(scope): summary`. Keep PRs focused, link related issues, list the commands you ran, and attach screenshots for changed plots or regression error visualizations. Do not commit `logs/`, `__pycache__/`, secrets from `.env`, or large generated datasets unless the PR explicitly requires them.
