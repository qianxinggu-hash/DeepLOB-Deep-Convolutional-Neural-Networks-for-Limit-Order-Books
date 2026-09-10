# 0906：基于未来成交或盘口的成交与撤单逻辑

本目录把 `0831` 报告所沿用的旧成交路径替换为明确的订单生命周期。定价、方向模型、walk-forward 划分和费用参数保持不变，便于只观察成交假设变化带来的影响。

完整实验结论见 [0906 成交逻辑实验报告](0906_成交逻辑实验报告.md)；报告源数据为 [固定参数成交逻辑诊断](output/fill_logic_diagnostic.json)和[完整 walk-forward 结果](output/july_2026_tolerant_as_gp_drift_results.json)。

## 成交规则

在 snapshot `t` 产生报价后：

1. 订单在 `t` 的时间戳加 5ms 后生效；更早的盘口不能造成成交。
2. 只检查后续 `t+1 ... t+20` 对应生命周期内的 L1 盘口和逐笔成交，不使用 `t` 当下证据。
3. 买单在 `bid quote >= future best ask` **或** `bid quote >= future trade price` 时成交。
4. 卖单在 `ask quote <= future best bid` **或** `ask quote <= future trade price` 时成交。
5. `touch` 允许价格相等；`through` 要求在任一证据上再穿过一个 tick（HKD 0.02）。
6. `t+20` 仍属于有效成交窗口；检查完该 snapshot 后，未成交订单撤单。

每一侧订单最终必须满足 `报价数 = 成交数 + 到期撤单数`。结果记录
`trade_price_fills`、`best_quote_fills`（并拆分为 `best_ask_fills` 与
`best_bid_fills`）以及各自占全部 maker 成交的比例。每笔成交按最早触发的
证据互斥归类；两类证据时间戳相同时沿用撮合逻辑，归入盘口证据。

## 参数口径与成交逻辑

价格预测参数和成交参数应当分开：

- 标签阈值 `alpha`、方向幅度 `calibrated_alpha` 与区间方差 `sigma²` 应只基于严格因果对齐的逐笔成交价路径。
- `k` / `order_decay_per_tick` 描述不同报价距离下的成交强度，应随本目录的成交规则重新估计。
- `gamma`、库存上限、报价距离和 `eta` 若通过历史回测选择，会间接受成交规则影响，但不应反向改变 `alpha` 或 `sigma²` 的价格样本。

对 snapshot `t`，成交价定义为同一交易时段内、时间戳不晚于 `t` 的最近一笔成交；不得使用 `t` 之后的成交进行填充，也不得跨午休等时段继承。20-snapshot 目标及方差统一由

```text
trade_price[t] = last trade price with trade_time <= snapshot_time[t]
future_trade_move_ticks[t]
  = (trade_price[t+20] - trade_price[t]) / tick_size
```

产生。当前 `0906` 结果只完成了成交生命周期改造，沿用的 `alpha/sigma²` 仍来自旧版 L1 中价口径；在完成 trade-price-only 改造并重新运行前，现有结果不能视为新参数口径的最终回测。

## 运行

```bash
python -m unittest discover -s 0906/tests -v
python 0906/run_book_fill_experiment.py --reconstruction-policy tolerant --workers 3
python 0906/run_gp_qvi_experiment.py
python 0906/build_gp_qvi_report_artifact.py
python 0906/package_gp_qvi_report.py
```

月度结果写入 `0906/output/`。程序复用 `0824/data/july_2026/` 的重建缓存；若缓存不存在，则仍按原流程从 `DEEPLOB_7709_DATA_DIR` 或默认下载目录重建。

## GP-QVI 复现

`gp_qvi_model.py` 按 Guilbaud–Pham 的线性效用形式，在（剩余时间、库存、spread 状态）上求解有限状态 Bellman/QVI；spread 转移、六个日内时段的 Poisson 时钟强度和成交强度均仅由测试日之前最多 5 个完整交日滚动估计。GP 回放使用本目录的 5ms / `t+1…t+20` 新成交逻辑，并按当日 L1 价格网格识别 HKD 0.05 或 0.02 的合法 tick。

主结果为 `output/july_2026_gp_qvi_new_fill_results.json`，逐日结果为 `output/july_2026_gp_qvi_new_fill_daily.csv`，可阅读报告为 `output/7709_202607_GP_QVI回测报告.html`。

## 边界

这是基于逐笔触价或 L1 对手盘穿越的可执行性代理，不等同于交易所撮合回放。数据仍不包含虚拟订单的 queue-ahead、隐藏流动性，以及下单/撤单确认，因此不能证明触价时的真实排队成交。
