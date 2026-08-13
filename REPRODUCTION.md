# DeepLOB 本地复现记录

## 当前状态

- 官方仓库已克隆到本目录。
- 官方 FI-2010 数据已下载并解压。
- 已在 Python 3.12.10、PyTorch 2.13.0 CPU 环境中完成训练与测试。
- 已完成官方设置的 50 epoch 全量 CPU 实验，并保存验证集最优 checkpoint。

Windows 未启用 Long Path，而仓库目录名较长，PyTorch 无法安装到仓库内的 `.venv`。可用环境位于：

```powershell
C:\dlobenv
```

## 复现脚本

`reproduce_pytorch.py` 保留了官方 notebook 的以下设置：

- FI-2010 Decimal Precision、No Auction 数据；
- 前七天数据的 80%/20% 训练与验证切分，后三天作为测试；
- 40 个十档 LOB 特征；
- 长度为 100 个事件的输入窗口；
- 第五组标签，即预测 horizon 10；
- CNN + Inception + LSTM 架构，共 143,907 个参数；
- batch size 64、Adam 学习率 0.0001；
- 与官方 notebook 一样，将 softmax 概率交给 `CrossEntropyLoss`。

脚本使用懒加载窗口，避免官方 notebook 预先生成全部重叠窗口所需要的大量内存。

## 已完成的运行

快速连通性测试：

```powershell
C:\dlobenv\Scripts\python.exe reproduce_pytorch.py `
  --epochs 1 `
  --max-train-samples 2048 `
  --max-val-samples 512 `
  --max-test-samples 512 `
  --output-dir artifacts\smoke
```

结果：训练、验证、模型保存、重新加载和测试均成功。

CPU 缩小版实验：

```powershell
C:\dlobenv\Scripts\python.exe reproduce_pytorch.py `
  --epochs 5 `
  --max-train-samples 20000 `
  --max-val-samples 5000 `
  --max-test-samples 5000 `
  --output-dir artifacts\reduced
```

结果：

- 每个 epoch 约 80–88 秒；
- 最低 validation loss：1.1130；
- 测试 accuracy：0.5364；
- 完整指标保存在 `artifacts/reduced/reproduction_results.json`。

该结果只证明本地训练流程和模型实现可运行。由于只取每个集合开头的一小段并仅训练 5 个 epoch，不能与官方 notebook 的全量 50 epoch 测试 accuracy 0.7535 直接比较。

## 全量 CPU 复现结果

```powershell
C:\dlobenv\Scripts\python.exe reproduce_pytorch.py `
  --epochs 50 `
  --output-dir checkpoints\full_cpu
```

实际运行约 10 小时 44 分钟。数据规模为 203,701 个训练窗口、50,851 个验证窗口和 139,488 个测试窗口。

- 最低 validation loss：0.8993（epoch 20）；
- 测试 accuracy：0.7383；
- macro F1：0.7384；
- weighted F1：0.7392；
- 官方 notebook 中记录的 accuracy 为 0.7535，本次结果低 1.52 个百分点；
- 多数类基线 accuracy 约为 0.3445。

验证集在 epoch 20 后不再改善，而训练损失继续下降，说明后半段出现过拟合。测试使用的是 epoch 20 保存的最优模型，不是 epoch 50 的最终模型。

产物：

- `checkpoints/full_cpu/best_deeplob_state.pt`：614,121 bytes；
- `checkpoints/full_cpu/reproduction_results.json`：完整训练历史和分类指标；
- checkpoint SHA-256：`f226acbb85377d2d4036f90eb1415b6daae8f779c4a80274216a3d10cbc2d44e`。

## 依赖

若要重新创建短路径环境：

```powershell
python -m venv C:\dlobenv
C:\dlobenv\Scripts\python.exe -m pip install -r requirements-reproduction.txt
```

## 迁移到 07709 数据前还需要做什么

FI-2010 的每个时点已经是标准化后的十档 LOB 快照和未来方向标签，而 `hk07709_2026-07-09.csv` 是 Add/Modify/Delete/Trade 消息流。不能直接把 CSV 的各列送入 DeepLOB。需要先：

1. 按消息顺序重建十档 bid/ask 价格与数量；
2. 每个事件生成 40 维 `[price, volume] × 10 levels × 2 sides` 快照；
3. 仅用训练期统计量做标准化；
4. 依据未来 mid-price 构造 up/stationary/down 标签；
5. 按交易日切分训练、验证、测试，避免重叠窗口造成数据泄漏。
