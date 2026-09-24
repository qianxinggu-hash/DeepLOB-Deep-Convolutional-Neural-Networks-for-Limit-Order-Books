#!/usr/bin/env python3
"""Build the concise, three-question direction and market-making report."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRED = HERE / "output/aligned_direction_comparison/results.json"
REPLAY = HERE / "output/aligned_direction_market_making_hard_one_tick/results.json"
VALIDATION = HERE / "output/aligned_direction_market_making_hard_one_tick/validation.json"
OLD_REPLAY = HERE / "output/aligned_direction_market_making/results.json"
CLASS_DISTRIBUTION = (HERE /
                      "output/aligned_direction_comparison/hard_one_tick_class_distribution.json")
OUT = HERE.parent / "0920report/README.md"
EVIDENCE = OUT.parent / "evidence"
MODEL_NAMES = {
    "deeplob": "DeepLOB",
    "old_hybrid": "旧 Hybrid（+L2）",
    "new_l3_hybrid": "New Hybrid（+L3）",
}


def net(value: float) -> str:
    return f"{value:,.2f}"


def hkd(value: float) -> str:
    return f"{'−' if value < 0 else '+'}HK${abs(value):,.2f}"


def main() -> None:
    p = json.loads(PRED.read_text())
    r = json.loads(REPLAY.read_text())
    v = json.loads(VALIDATION.read_text())
    old_r = json.loads(OLD_REPLAY.read_text())
    class_counts = json.loads(CLASS_DISTRIBUTION.read_text())
    ps = {(row["horizon"], row["subset"], row["model"]): row
          for row in p["summary"]}
    rs = {(row["family"], row["fill_mode"], row["gamma"],
           row["subset"], row["model"]): row for row in r["summary"]}
    old_rs = {(row["family"], row["fill_mode"], row["gamma"],
               row["subset"], row["model"]): row for row in old_r["summary"]}
    lines = [
        "# 7709：L3 信号、涨跌预测和做市收益",
        "",
        "_2026 年 7 月 19 个逐日测试日。三组模型使用相同的未来中价涨／平／跌标签、报价起点和冻结 DeepLOB 编码器；每天用此前最多五个完整日重训分类头。这里比较的是新增 L2、L3 信息的效果，三套 CNN 尚未分别端到端重训。L3 只在 10 个订单状态完整的测试日使用，其余 9 日严格回退到旧 Hybrid。_",
        "",
        "## 1. 加入了哪些 L3 信号？",
        "",
        "六个信号都从逐笔 OrderID 生命周期重建，描述买卖最优价两侧的差异：",
        "",
        "1. 最优价**订单数量不平衡**：买一与卖一由多少笔独立订单组成。",
        "2. 最优价**订单年龄差**：两侧挂单的数量加权平均存活时间。",
        "3. 最优价**新订单占比差**：挂出不超过 5 秒的量占各侧挂单量的比例。",
        "4. 最优价**订单大小集中度差**：挂单量是否集中在少数大订单。",
        "5. 近 5 秒**老订单移除量差**：存活至少 10 秒的订单被移除多少量。",
        "6. 近 5 秒**订单移除次数差**：两侧有多少次订单移除事件。",
        "",
        "单独使用时，订单数量不平衡和订单移除次数差最有效。[特征定义](../0920/l3_order_signal.py)、[单因子结果](evidence/l3_ablation.json)",
        "",
        "## 2. 预测效果提高了多少？",
        "",
        "下表是**所有报价**的涨／平／跌三分类准确率，三种输入使用完全相同的标签与测试报价。[完整预测结果](evidence/prediction_results.json)",
        "",
        "| 模型 | 未来 5 snapshot | 未来 20 snapshot |",
        "| --- | ---: | ---: |",
    ]
    for model, label in MODEL_NAMES.items():
        lines.append(f"| {label} | {100*ps[(5, 'all', model)]['accuracy_3class']:.2f}% | {100*ps[(20, 'all', model)]['accuracy_3class']:.2f}% |")
    five = v["prediction_day_bootstrap"]["h5_l3_available"]["new_minus_old"]
    twenty = v["prediction_day_bootstrap"]["h20_l3_available"]["new_minus_old"]
    lines += [
        "",
        f"在 **10 个完整 L3 日**直接配对，加入 L3 使 5-snapshot 准确率提高 **{five['delta_percentage_points']:.2f} 个百分点**；20-snapshot 提高 **{twenty['delta_percentage_points']:.2f} 个百分点**，后者的交易日重抽样区间跨过零。整体提升主要来自旧 Hybrid 加入 L2 工程特征，L3 贡献较小。[配对验证](evidence/hard_validation.json)",
        "",
        "## 3. 预测提升转化成做市收益了吗？",
        "",
        "对 **20-snapshot** 三分类直接取概率最高的类别：涨 = **+1 tick**、平 = **0 tick**、跌 = **−1 tick**，不再拟合概率到 tick 的幅度。把它接入相同的 AS 和 GP/QVI 报价器。下表是**整月净收益，港元，已扣费用**；不交易为 HK$0。through 与 touch 是两种代理成交条件。[回放明细](evidence/hard_replay_results.json)",
        "",
        "| 做市方法 | 成交代理 | γ | 无 drift | DeepLOB | 旧 Hybrid | New Hybrid |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in ("AS", "GP"):
        for mode in ("through", "touch"):
            for gamma in ((None,) if family == "AS" else (5.0, 3.125)):
                values = [rs[(family, mode, gamma, "all", model)]["net_hkd"]
                          for model in ("no_drift", *MODEL_NAMES)]
                lines.append(f"| {family} | {mode} | {'—' if gamma is None else gamma} | "
                             + " | ".join(net(x) for x in values) + " |")
    hard_through = rs[("GP", "through", 5.0, "all", "new_l3_hybrid")]
    hard_touch = rs[("GP", "touch", 5.0, "all", "new_l3_hybrid")]
    calibrated_through = old_rs[("GP", "through", 5.0, "all", "new_l3_hybrid")]
    calibrated_touch = old_rs[("GP", "touch", 5.0, "all", "new_l3_hybrid")]
    flat = class_counts["20_new_l3_hybrid"]["0"]
    total = class_counts["20_new_l3_hybrid"]["total"]
    lines += [
        "",
        f"**没有转化为盈利。** 与上一版概率校准映射相比，New Hybrid 的 GP（γ=5）through 净收益从 **{hkd(calibrated_through['net_hkd'])}** 变为 **{hkd(hard_through['net_hkd'])}**，touch 从 **{hkd(calibrated_touch['net_hkd'])}** 变为 **{hkd(hard_touch['net_hkd'])}**。touch 的 maker 成交由 **{calibrated_touch['maker_fills']:,}** 增至 **{hard_touch['maker_fills']:,}**，费用由 **HK${net(calibrated_touch['fees_hkd'])}** 增至 **HK${net(hard_touch['fees_hkd'])}**。[旧映射结果](evidence/calibrated_replay_results.json)、[直接映射回放对账](evidence/hard_validation.json)",
        "",
        f"原因之一是 20-snapshot New Hybrid 只在 **{100*flat/total:.2f}%** 的报价上把‘平’选为最高概率类，直接映射后几乎总给出 ±1 tick。AS-through 虽比此前校准映射少亏，仍亏逾 HK$30 万；两种成交代理下所有列出的策略都没有超过不交易的 HK$0。[类别分布](evidence/class_distribution.json)",
        "",
        "7 月数据已被用于多轮方法探索，且回放没有真实排队位置。当前结果支持 L3 含有弱方向信息，不支持启用 L3 drift 报价。",
        "",
    ]
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for source, name in (
        (PRED, "prediction_results.json"),
        (HERE / "output/direction_l3_ablation/results.json", "l3_ablation.json"),
        (REPLAY, "hard_replay_results.json"),
        (VALIDATION, "hard_validation.json"),
        (OLD_REPLAY, "calibrated_replay_results.json"),
        (CLASS_DISTRIBUTION, "class_distribution.json"),
    ):
        shutil.copy2(source, EVIDENCE / name)
    OUT.write_text("\n".join(lines))
    print(OUT)


if __name__ == "__main__":
    main()
