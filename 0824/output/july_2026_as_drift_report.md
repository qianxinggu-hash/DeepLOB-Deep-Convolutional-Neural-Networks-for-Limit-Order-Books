# HK 07709：2026 年 7 月 AS + drift 月度回测

## 模型

AS+drift 保留经典 Avellaneda–Stoikov 的库存惩罚与最优 spread，只在 reservation price 中加入
校准后的方向预期：

```text
r = mid + eta * calibrated_alpha - inventory * gamma * horizon_variance
total_spread = gamma * horizon_variance
             + 2 / gamma * log(1 + gamma / order_decay)
```

本实验区分两种实现：

- `AS + unit drift`：沿用当天经典 AS 的全部参数，固定 `eta=1`；这是 drift 本身最干净的配对。
- `AS + tuned drift`：在历史验证期联合选择 gamma、库存上限和 eta。

另报告联合调参版强制 `eta=0` 的配对消融，以及只有验证期胜过经典 AS 才启用的 kill switch。

## 全费用结果

| 成交口径 | 模型 | 净 PnL | 净 PnL/计费周转 | 盈利日 | 相对经典 AS |
|---|---|---:|---:|---:|---:|
| through | 经典 AS | **-1,029.57** | **-0.108 bp** | 4/10 | — |
| through | AS + unit drift | -1,269.82 | -0.132 bp | 4/10 | -240.25 |
| through | AS + tuned drift | -7,082.53 | -0.632 bp | 4/10 | -6,052.96 |
| through | tuned 参数、关闭 drift | -1,896.29 | -0.202 bp | 4/10 | -866.72 |
| through | AS + drift kill switch | -9,004.08 | -0.820 bp | 4/10 | -7,974.51 |
| touch | 经典 AS | +12,879.76 | +0.917 bp | 7/10 | — |
| touch | AS + unit drift | **+13,063.30** | **+0.923 bp** | 7/10 | +183.54 |
| touch | AS + tuned drift | +9,676.98 | +0.609 bp | 7/10 | -3,202.78 |
| touch | tuned 参数、关闭 drift | **+15,990.23** | **+1.158 bp** | 7/10 | +3,110.47 |
| touch | AS + drift kill switch | +4,163.49 | +0.267 bp | 7/10 | -8,716.28 |

## 解释

固定 `eta=1` 的纯 AS+drift 在两种排队口径下都与经典 AS 非常接近：保守口径少
HKD 240，乐观口径多 HKD 184。相对于约 HKD 0.96–1.42 亿的计费周转，这个差异没有实际
经济显著性，也会随不可观测的排队成交假设翻转。

联合调参版的 drift 明显更差。保守口径下，它相对相同 gamma、库存上限但关闭 drift 的配对
消融少 HKD 5,186；乐观口径少 HKD 6,313。这说明亏损来自方向平移本身，而不只是 AS 参数
被重新选择。

kill switch 在 07-07、07-15、07-23、07-27 启用，但只在其中两个保守测试日胜过经典 AS。
07-07 的验证期选择尤其失效：经典 AS 当日净赚 HKD 13,315，而放大的 AS+drift 只赚
HKD 5,275。过去五个验证区间的正目标没有稳定预测下一日增量。

因此，当前最准确的结论是：**AS+drift 已实现并测试，但没有稳定改善经典 AS。** 方向模型
此前对 GP-style 基线的明显改善，不能外推到本身已经接近盈亏平衡的 AS 报价上。

## 验证与限制

- 18项单元测试通过，包括零 drift 与经典 AS 逐价完全相同、正 drift 上移 reservation price。
- 时序、费用、成交数、配对参数、kill switch、月度汇总和经典 AS 复现检查全部通过。
- 主结果仍只有10个通过严格 MBO 质量门槛的完整样本外日。
- `touch/through` 是队列位置敏感性，不是精确实盘成交。

完整结果见 `july_2026_as_drift_results.json`，逐日表见
`july_2026_as_drift_daily.csv`，核验结果见 `july_2026_as_drift_validation.json`。
