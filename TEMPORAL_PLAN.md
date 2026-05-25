# Temporal Extension Implementation Plan (DarkIR)

## Goal
Add a temporal attention fusion module (TFM) that takes a short frame window (T=3) and improves low-light video restoration over the single-frame baseline, with minimal disruption to the existing DarkIR backbone.

## Scope (Stage 1: Minimal, stable, high-impact)
- Keep the DarkIR backbone unchanged.
- Insert TFM right after the intro conv and before Encoder stage 1.
- Use T=3 (center frame is the target output).
- Use BVI-Lowlight (aligned video dataset) for temporal training and evaluation.

## Architecture Change (Where to insert)
Current (single frame):
`input -> intro conv -> Encoder stage 1 -> ... -> Decoder -> output`

Temporal (T=3):
`input [B,T,C,H,W] -> intro conv (per-frame) -> TFM -> Encoder stage 1 -> ... -> Decoder -> output (center frame)`

## TFM (Temporal Attention, conservative)
- Inputs: per-frame features `[B,T,F,H,W]` from intro conv.
- Compute attention weights per time step (no spatial mixing):
  - Global average pool over H,W to get `[B,T,F]`.
  - MLP or 1x1 Conv over F to get logits per time.
  - Softmax over T to get weights.
- Fuse: weighted sum across T to `[B,F,H,W]`.
- Residual gate to protect stability:
  - `fused = center + alpha * (fused - center)` with small `alpha` (e.g., 0.1 init).

## Data Pipeline (BVI-Lowlight video windows)
- Scene layout: `S01_colour_chart/low_light_10`, `low_light_20`, `normal_light_10`, `normal_light_20`, each with frames like `00248.png`.
- Split: random **32 train / 8 test** scenes (fixed seed), saved to a split list for reproducibility.
- Add a video dataset loader that returns a window of T consecutive frames and the center target frame.
- Expected shape: input `[B,T,C,H,W]`, target `[B,C,H,W]`.
- Use `frame_stride=1` with boundary replication at clip ends.

## Training Losses
- Keep existing image reconstruction losses (pixel/perceptual/edge as currently used in DarkIR).
- Add temporal consistency loss only if PSNR/SSIM gains are insufficient; start with image losses only for stability.

## Inference (Video)
- Modify `inference_video.py`:
  - Maintain a sliding window of T frames.
  - For each frame index i, feed frames `[i-1, i, i+1]` and output the center frame.
  - At boundaries, replicate the first/last frame.
- Output video should still be side-by-side (original + restored) unless changed.

## Configs to Add
- `options/train/Temporal_BVI.yml` (new):
  - dataset path, temporal window, stride, batch size.
  - model settings (same as baseline).
  - loss weights (temporal consistency optional).

## Files Likely to Change
- `archs/` (add TFM module, update model init for temporal input)
- `data/` (add temporal dataset loader)
- `options/` (add temporal training config)
- `inference_video.py` (use temporal window instead of single-frame)
- `testing_unpaired.py` or add a temporal testing script for video datasets

## Verification
- Train baseline single-frame model on same dataset/split for fair comparison.
- Evaluate on BVI-Lowlight test scenes:
  - PSNR/SSIM
- Visual check: reduced flicker and better noise suppression in dark regions.

## Stage 2 (Optional if Stage 1 gains are small)
- Add another TFM after Encoder stage 1.
- Increase window to T=5.
- Consider lightweight optical-flow alignment if misalignment artifacts appear.
