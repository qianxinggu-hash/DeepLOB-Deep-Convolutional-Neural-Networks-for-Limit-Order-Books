# 0824：HK 07709 单日重建与 DeepLOB 70/30 实验

本目录针对“逐笔事件恢复盘口可能错误、跨日不连续”的问题建立独立实验。它先按
`MsgType=50` 的成交数量选择成交量最高交易日，再从该日的 MBP/L2 绝对档位事件流
恢复十档盘口，按时间严格 70%/30% 切分并训练 DeepLOB。

原始逐笔数据、重建缓存、模型权重和逐成交输出不包含在 Git 仓库中。默认从仓库根目录下的
`7709_tickdata/` 读取原始文件，也可以通过环境变量指定数据目录：

```powershell
$env:DEEPLOB_7709_DATA_DIR = 'D:\path\to\7709_tickdata'
```

脚本会在本地 `0824/data/`、`0824/checkpoints/` 和 `0824/output/` 生成缓存与结果；这些目录
已被 `.gitignore` 排除。论文式结果摘要保留在 `project_methodology.md` 和
`reconstruction_and_results.md` 中。

运行：

```powershell
python 0824/run_experiment.py --device cuda
```

测试：

```powershell
python -m unittest discover 0824/tests -v
```

实验完成后，指标在 `output/results.json`，权重在 `checkpoints/`，详细恢复方法、异常
处理和结论见 `reconstruction_and_results.md`。

针对初版 DeepLOB 过拟合的改进实验：

```powershell
python 0824/run_improved_experiment.py
python 0824/validate_improved_results.py
```

改进结果在 `output/improved_results.json`。该版本在前70%内部再做严格时间验证，选择
标签 horizon、正则化模型、早停轮数和融合权重，然后在全部前70%重训，最后评估后30%。

多日与策略检验：

```powershell
python 0824/run_multiday_strategy.py
python -m jupyter nbconvert --execute --to notebook --inplace 0824/multiday_strategy_analysis.ipynb
python 0824/validate_multiday_strategy.py
```

该实验使用严格 MBO 重放检查重新上传的07-09，并将 MBO 与 MBP 口径分开做逐日70/30和
跨日 walk-forward。回测在下一快照以买一/卖一成交、禁止重叠持仓、测试多档费用，并额外
比较 PnL 对齐标签和20/50/100档持有期。结论、费用依据和异常隔离规则见
`reconstruction_and_results.md` 第11节；可复现 notebook 为
`multiday_strategy_analysis.ipynb`。

方向增强做市实验：

```powershell
python 0824/run_directional_market_maker.py
python 0824/validate_directional_market_maker.py
python -m jupyter nbconvert --execute --to notebook --inplace 0824/directional_market_making_analysis.ipynb
```

该模型把跨日方向概率作为 AS 保留价的条件漂移，使用 GP 风格的离散 tick、库存上限和
日末主动平仓；同时设置验证期 alpha kill switch。原始成交打印只支持 touch/through
排队敏感性上下界，因此结果可以用于研究设计，不能视为精确实盘成交。完整公式、两次
walk-forward 结果、费用分解与结论见 `reconstruction_and_results.md` 第12节。

秒级 markout、双侧逆向选择和配对增量 PnL：

```powershell
python 0824/run_directional_markout.py
python 0824/validate_directional_markout.py
python -m jupyter nbconvert --execute --to notebook --inplace 0824/directional_markout_analysis.ipynb
```

该实验分别使用 MBO 和 MBP 两条 walk-forward 链，计算每笔模拟 maker fill 后5/10/20秒
markout，分别报告买单和卖单 adverse-selection cost，并与相同参数、关闭方向信号的策略
配对比较。术语、结果和连续日数据限制见 `reconstruction_and_results.md` 第13节。

项目完整方法论总结见 [`project_methodology.md`](project_methodology.md)。该文档按
DeepLOB、25维因果逻辑回归、方向信号校准和 AS/GP-style 做市的顺序，统一说明公式、
数学原理、实证结果、证据边界与复现入口。
