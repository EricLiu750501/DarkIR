# DarkIR agent notes

## Commands (verified entrypoints)
- Install deps: `pip install -r requirements.txt`
- Evaluate paired datasets: `python testing.py -p ./options/test/<config.yml>` (defaults to `./options/test/LOLBlur.yml`)
- Evaluate unpaired datasets: `python testing_unpaired.py -p ./options/test/<config.yml>` (default `./options/test/RealBlur_Night.yml`)
- Inference on folder: `python inference.py -i <folder_path>` (uses `./options/inference/LOLBlur.yml` unless overridden)
- Inference on video: `python inference_video.py -i /path/to/video.mp4` (uses `./options/inference_video/Baseline.yml` unless overridden)

## Data, weights, outputs
- Test datasets are expected under `./data/datasets/...` (see `options/test/*.yml` for exact paths like `./data/datasets/LOLBlur/test`).
- Model weights are read from `./models/...` per `options/*/*.yml` (e.g., `./models/DarkIR_384.pt`, `./models/DarkIR_64width.pt`).
- Inference outputs go to `./images/results` (folder inference) and `./videos/results` (video inference).

## Config gotchas
- Entry scripts set `CUDA_VISIBLE_DEVICES` internally (`testing.py`, `testing_unpaired.py`, `inference.py`, `inference_video.py`). If you need a different GPU, change it in those scripts before import torch.
- `app.py` references `./options/predict/LOLBlur.yml`, but there is no `options/predict/` in this repo; add or update the path before running the Gradio app.
