# 20-snapshot 三分类平衡阈值实验

## 做法

沿用 `run_aligned_direction_comparison.py` 的冻结 DeepLOB 编码器、相同的报价起点、最多五个此前完整 L3 交易日，以及相同的 `StandardScaler + LogisticRegression(C=0.05)` 分类头。每个测试日只用训练日的未来 20-snapshot 中价变动选择一个对称阈值：`move < -θ` 为跌，`|move| ≤ θ` 为平，`move > θ` 为涨。候选阈值在半 tick 网格上，选择训练集中三类占比最接近各三分之一的阈值；该日测试标签与训练标签使用同一个 θ。测试日不参与选阈值。

19 个测试日中，前 3 日选中 `θ=0.5 tick`，其余 16 日为 `θ=1.0 tick`。这是滚动阈值实验，并非把测试集占比直接调到三分之一。

## 样本外结果

2026 年 7 月 19 个测试日、99,394 个非重叠报价起点。三类实际占比为跌 **34.83%**、平 **31.52%**、涨 **33.65%**。

| 模型 | 准确率 | 平衡准确率 | Macro F1 | Log loss | 最高概率最大值 | `max(p)>0.9` 次数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 冻结 DeepLOB 特征 | 38.62% | 38.87% | 38.48% | 1.0855 | 0.739 | 0 |
| 旧 Hybrid（+L2） | 40.13% | 40.41% | 39.96% | 1.0776 | 0.864 | 0 |
| 新 Hybrid（+L3） | **40.27%** | **40.52%** | **40.13%** | **1.0770** | **0.875** | **0** |

使用每个测试日的训练集多数类作为固定预测，准确率为 **37.14%**；使用训练集类别比例作为固定概率，log loss 为 **1.0918**。新 Hybrid 相对这两个因果基线均有小幅改善。按交易日重抽样的准确率增量区间约为 **+1.6 至 +5.0 个百分点**；该区间只衡量 19 日间的波动，不覆盖本项目反复研究 7 月数据造成的选择偏差。

新 Hybrid 对“平”的召回率为 **50.49%**，明显高于旧的零变动标签实验；但这个比较更换了标签定义，不能把两个实验的原始准确率直接相减作为模型提升。三类方向仍难预测，且没有出现 0.9 以上概率。因此，平衡标签解决了“平类只有约 8%”的标签比例问题，没有解决 0.9 置信度门槛下无信号的问题。

## 过拟合判断与边界

该实验重训的是**冻结编码器后的分类头**，没有端到端重训 DeepLOB。分类头按日加权的训练准确率分别为 DeepLOB 特征 **39.85%**、旧 Hybrid **41.75%**、新 Hybrid **41.88%**；对应样本外为 **38.62%**、**40.13%**、**40.27%**。这组差距约 1–2 个百分点，没有显示分类头严重记忆训练集；整体预测能力仍较弱。

此前另一组端到端 DeepLOB 训练在不同标签和日期上，训练/验证/测试准确率分别为 **35.88% / 34.74% / 30.93%**；此前端到端 Hybrid 分别为 **61.79% / 59.25% / 55.66%**。它们显示跨日期泛化下降，但不能与本实验的 40.27% 直接比较，也不能仅凭当前概率没有超过 0.9 就断定过拟合。参见 `output/results/7709_deeplob_chronological.json` 和 `output/results/7709_hybrid_deeplob.json`。

阈值与分类头都使用此前日期，但 7 月数据已经被用于多轮方法探索，结果仍需新日期验证。

## 与报价回放的连接

`run_aligned_direction_comparison.py --label-mode balanced` 将同一套逐日滚动阈值用于冻结 DeepLOB、旧 Hybrid、新 L3 Hybrid 的分类头，并导出与原先相同报价起点对齐的 `-1/0/+1 tick` 信号。用这些信号回放 19 个测试日，GP 取 `gamma=5`，包括手续费；`through` 和 `touch` 都只是成交代理。下表为合计净收益，单位 HKD：

| 策略 / 成交代理 | 无 drift | 旧 Hybrid：原标签 → 平衡标签 | 新 L3 Hybrid：原标签 → 平衡标签 |
| --- | ---: | ---: | ---: |
| GP / through | -119.54 | -1,572.23 → -1,869.40 | -1,497.19 → -1,743.80 |
| GP / touch | -2,439.69 | -14,412.05 → -17,422.46 | -14,310.30 → -17,323.96 |
| AS / through | -336,870.16 | -328,508.23 → -332,454.10 | -330,914.43 → -330,866.81 |
| AS / touch | -216,523.89 | -220,151.53 → -215,713.46 | -221,282.71 → -217,342.79 |

平衡标签改善了三分类定义，但没有证明报价策略获利；GP 两种成交代理下都变差。上述回放沿用每 20 个 snapshot 才开始一次报价的频率，**没有**测试每个 snapshot 都决策，也没有测试 `max(p)>0.9` 后才报价的收益。在已有 99,394 个报价起点上，三种模型都没有一次通过 0.9 门槛。

若把置信门槛改为严格 `max(p)>0.8`，冻结 DeepLOB 为 **0** 次、旧 Hybrid 为 **14** 次、新 L3 Hybrid 为 **18** 次。旧 Hybrid 的 14 次中有 7 次预测“平”，剩下 7 次方向候选的标签方向判对 4 次；新 L3 Hybrid 的 18 次中有 8 次预测“平”，剩下 10 次方向候选判对 4 次。样本太少，不能据此估算实盘收益。这个统计仍只覆盖每 20 个 snapshot 的报价起点。

门槛降至严格 `max(p)>0.5` 时，冻结 DeepLOB / 旧 Hybrid / 新 L3 Hybrid 分别有 **7,293 / 15,708 / 15,963** 次超过门槛；其中预测涨跌的方向候选分别为 **1,576 / 8,099 / 8,427** 次，占全部报价起点约 **1.59% / 8.15% / 8.48%**。方向候选按新标签判对 **640 / 3,596 / 3,739** 次，即 **40.61% / 44.40% / 44.37%**。新 L3 Hybrid 的 8,427 次方向候选中，4,085 次集中在 2026-07-07；频率有明显日间波动。这个门槛尚未进行单独的 GP/AS 收益回放，也未评估逐 snapshot 决策。

## 信号接口及实盘边界

`hybrid_signal_bundle.py` 用最近五个此前完整 L3 日训练三种分类头和共同标签阈值，并把阈值、类别顺序、冻结编码器的 checkpoint 哈希和模型保存在一个 bundle 中。`score_batch()` 接受每个 snapshot 的因果特征，输出三类概率、预测类别、最大概率与可配置置信门槛结果；L3 缺失时，新 L3 Hybrid 精确退回旧 Hybrid。三个头可在同一批 snapshot 上比较。评分时可传 `confidence_threshold=0.8`，无需重训分类头。

其他训练脚本中，`0824/run_experiment.py` 和 `train_7709_hybrid_deeplob.py` 原本已在训练样本上用 `|return|` 的三分之一分位数设置平类阈值；这与本实验“由训练日确定平盘区间”的原则一致，但它们的收益期限、标签单位和网络训练方式不同，不能复用本次的 0.5/1.0 tick 数值。`train_7709_hybrid_deeplob.py` 现已让 `--stationary-share` 实际控制该分位数，默认值仍为三分之一。`0920/build_new_hybrid_signal.py` 是连续 tick 回归，没有涨跌平标签可改。

这里的 `l2` 必须是已经计算好的 64 维 DeepLOB embedding 加 28 维因果 L2 特征，`l3` 是六维 OrderID 生命周期特征。仓库尚无实时行情特征流、报单/撤单接口、订单状态同步；本信号数据集的最后一天为 2026-07-30。这个 bundle 只能作为待接入的信号和影子评分接口，**不是可以运行的实盘交易系统**。`directional_quote_candidate` 仅表示方向类且最大概率超过当前门槛，不是报单指令。逐 snapshot 门槛通过率尚未测量。

## 复跑

从仓库根目录运行：

```bash
.venv/bin/python 0920/run_balanced_flat_label_experiment.py
.venv/bin/python 0920/run_aligned_direction_comparison.py --label-mode balanced
.venv/bin/python 0920/replay_aligned_direction_market_making.py \
  --signals-dir 0920/output/aligned_direction_comparison_balanced/signals \
  --output-dir 0920/output/aligned_direction_market_making_balanced_hard_gamma5 \
  --tick-map hard_one_tick --gammas 5
.venv/bin/python 0920/hybrid_signal_bundle.py train \
  --available-through 2026-07-30 \
  --bundle 0920/output/hybrid_signal_bundle/balanced_h20_through_2026-07-30.joblib
```

机器结果：`0920/output/balanced_flat_labels/results.json`；逐日概率和标签：`0920/output/balanced_flat_labels/per_day/`；回放结果：`0920/output/aligned_direction_market_making_balanced_hard_gamma5/results.json`。
