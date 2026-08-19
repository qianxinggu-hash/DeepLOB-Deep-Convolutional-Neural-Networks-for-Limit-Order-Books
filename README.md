# DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
This jupyter notebook is used to demonstrate our recent work, "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books", published in IEEE Transactions on Singal Processing. We use FI-2010 dataset and present how model architecture is constructed here. The FI-2010 is publicly available and interested readers can check out their paper.

Both tensorflow (version 1 and 2) and pytorch are available.

## HK 07709 chronological experiment

`train_7709_deeplob.py` reconstructs 10-level order books from the files in
`7709_tickdata/`, tunes the DeepLOB prediction horizon on 2026-08-04, and
holds 2026-08-07 out for the final test. The input lookback is 100 snapshots;
the candidate label horizons are 20, 50, and 100 snapshots.

```bash
uv run --python 3.12 --with numpy --with torch --with sortedcontainers \
  python train_7709_deeplob.py \
  --epochs 12 --patience 3 --batch-size 256 \
  --train-target-stride 10 --validation-target-stride 5 --device mps
```

Use `--device auto` on non-Apple machines. The machine-readable metrics are
written to `output/results/7709_deeplob_chronological.json`, and the selected
checkpoint is stored under `checkpoints/7709/`.

The original dynamic-price normalisation can hide short-horizon price changes
when the available dates are non-consecutive. The causal baseline and improved
Hybrid DeepLOB experiments diagnose and address that issue:

```bash
uv run --python 3.12 analyze_7709_causal_baselines.py

uv run --python 3.12 --with numpy --with torch --with sortedcontainers \
  python train_7709_hybrid_deeplob.py \
  --epochs 15 --patience 4 --train-stride 5 \
  --validation-stride 5 --batch-size 256 --device mps
```

The hybrid model anchors the full 100-state price path to the mid-price at the
prediction timestamp, retains the DeepLOB convolution/LSTM branch, and adds a
small branch containing causal lagged-return and depth-imbalance features.
