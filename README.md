# DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
This jupyter notebook is used to demonstrate our recent work, "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books", published in IEEE Transactions on Singal Processing. We use FI-2010 dataset and present how model architecture is constructed here. The FI-2010 is publicly available and interested readers can check out their paper.

Both tensorflow (version 1 and 2) and pytorch are available.

The latest 7709 L3 direction-signal and market-making comparison is in
[0920report](0920report/README.md).

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

## Using Hybrid DeepLOB as AS/GP drift

`hybrid_drift_adapter.py` converts the Hybrid DeepLOB down/stationary/up
probabilities into a calibrated expected move in ticks.  Calibration targets
must precede the replay period.  The expected move can be passed directly to
`as_drift_quote_prices`, whose reservation price is

```text
r = mid + expected_move - inventory * gamma * horizon_variance.
```

For the paper-style Guilbaud--Pham QVI, `0906/gp_qvi_model.py` provides
`solve_gp_drift_policy`.  It augments the original `(time, spread, inventory)`
state with a finite DeepLOB drift state and adds

```text
inventory * lot_size * drift_hkd_per_second * dt
```

to the Bellman running reward.  The learned drift-state transition matrix lets
the short-horizon signal decay or reverse instead of assuming that its current
value persists over the entire GP horizon.  `simulate_gp_day` accepts these
policies together with one causal drift-state index per quote decision.

The training LOB stores prices and sizes in normalized reconstruction units,
whereas the execution replay uses HKD and shares.  Keep the neural output in
ticks until the execution instrument's tick size is known; only then convert
ticks to HKD or HKD/second.

Run the chronological integration check with:

```bash
uv run --python 3.12 --with numpy --with scipy --with torch \
  --with sortedcontainers --with scikit-learn \
  python 0906/run_hybrid_drift_integration.py --device auto
```

The default split is 2026-08-04 calibration followed by 2026-08-07 test.
`--gp-gamma` is exposed for pre-test tuning; it must not be selected using the
reported test-day PnL.
