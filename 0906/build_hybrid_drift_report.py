#!/usr/bin/env python3
"""Build a source-backed report for the Hybrid DeepLOB + AS/GP experiment."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "0906" / "output"
DEFAULT_RESULT = OUTPUT / "hybrid_deeplob_as_gp_integration.json"
GAMMA_RESULT = OUTPUT / "hybrid_deeplob_as_gp_gamma_0_1_diagnostic.json"
HYBRID_RESULT = ROOT / "output" / "results" / "7709_hybrid_deeplob.json"
ARTIFACT = OUTPUT / "hybrid_drift_detailed_report_artifact.json"
TITLE = "Hybrid DeepLOB Drift 与 AS / GP-QVI 详细结果"


def source(source_id: str, label: str, path: str, description: str) -> dict:
    item = {"id": source_id, "label": label, "path": path, "description": description}
    if path.endswith(".json"):
        item["query"] = {
            "engine": "duckdb",
            "sql": f"SELECT * FROM read_json_auto('{path}')",
            "description": description,
            "id": f"{source_id}-json",
        }
    return item


def pct(value: float) -> float:
    return 100.0 * value


def result_row(model: str, gamma: float, fill_mode: str, strategy: str, row: dict) -> dict:
    actions = row.get("action_counts", {})
    return {
        "model": model,
        "gamma": gamma,
        "fill_mode": fill_mode,
        "strategy": strategy,
        "gross_pnl_hkd": row["gross_pnl_hkd"],
        "fees_hkd": row.get("fees_hkd", row.get("all_in_fees_hkd", 0.0)),
        "net_pnl_hkd": row["net_pnl_hkd"],
        "maker_fills": row["maker_fills"],
        "buy_fills": row["buy_fills"],
        "sell_fills": row["sell_fills"],
        "quoted_bid": row["quoted_bid"],
        "quoted_ask": row["quoted_ask"],
        "max_inventory_lots": row["max_abs_inventory_lots"],
        "max_drawdown_hkd": row["max_drawdown_hkd"],
        "feeable_turnover_hkd": row["feeable_turnover_hkd"],
        "net_bps": row.get("net_bps_of_maker_turnover", row.get("net_bps_of_feeable_turnover")),
        "market_order_lots": row.get("market_order_lots", 0),
        "bid_none": actions.get("bid_none"),
        "bid_best": actions.get("bid_best"),
        "bid_improve": actions.get("bid_improve"),
        "ask_none": actions.get("ask_none"),
        "ask_best": actions.get("ask_best"),
        "ask_improve": actions.get("ask_improve"),
    }


def delta_row(model: str, gamma: float, fill_mode: str, baseline: dict, drift: dict) -> dict:
    return {
        "model": model,
        "gamma": gamma,
        "scenario": f"{model} / {fill_mode}",
        "fill_mode": fill_mode,
        "net_delta_hkd": drift["net_pnl_hkd"] - baseline["net_pnl_hkd"],
        "gross_delta_hkd": drift["gross_pnl_hkd"] - baseline["gross_pnl_hkd"],
        "fee_delta_hkd": drift.get("fees_hkd", drift.get("all_in_fees_hkd", 0.0))
        - baseline.get("fees_hkd", baseline.get("all_in_fees_hkd", 0.0)),
        "fill_delta": drift["maker_fills"] - baseline["maker_fills"],
        "drawdown_delta_hkd": drift["max_drawdown_hkd"] - baseline["max_drawdown_hkd"],
    }


def main() -> None:
    default = json.loads(DEFAULT_RESULT.read_text(encoding="utf-8"))
    gamma = json.loads(GAMMA_RESULT.read_text(encoding="utf-8"))
    hybrid = json.loads(HYBRID_RESULT.read_text(encoding="utf-8"))
    generated_at = datetime.now().astimezone().isoformat()

    test = hybrid["test"]
    class_metrics = [
        {
            "class": row["class"],
            "precision_pct": pct(row["precision"]),
            "recall_pct": pct(row["recall"]),
            "f1_pct": pct(row["f1"]),
            "support": row["support"],
        }
        for row in test["per_class"]
    ]
    confusion = []
    labels = ["down", "stationary", "up"]
    for actual, values in zip(labels, test["confusion_matrix"]):
        confusion.append({"actual": actual, **{f"pred_{label}": value for label, value in zip(labels, values)}})

    signal = default["signal"]
    regime = signal["drift_regime"]
    state_counts = default["gp_results"]["touch"]["hybrid_drift_gp"]["drift_state_counts"]
    drift_states = []
    for index, (label, rate, transition) in enumerate(
        zip(regime["labels"], regime["drift_hkd_per_second"], regime["transition_probabilities"]), start=1
    ):
        count = state_counts.get(label, 0)
        drift_states.append(
            {
                "state": f"S{index}",
                "label": label,
                "drift_hkd_per_second": rate,
                "count": count,
                "share_pct": 100.0 * count / signal["test_quotes"],
                "self_transition_pct": 100.0 * transition[index - 1],
            }
        )

    as_rows = []
    gp_default_rows = []
    gp_gamma_rows = []
    deltas = []
    strategy_labels = {
        "classic_as": "Classic AS",
        "hybrid_drift_as": "Drift AS",
        "martingale_gp": "Martingale GP",
        "hybrid_drift_gp": "Drift GP",
    }
    for mode in ("through", "touch"):
        classic = default["as_results"][mode]["classic_as"]
        drift_as = default["as_results"][mode]["hybrid_drift_as"]
        for key, row in (("classic_as", classic), ("hybrid_drift_as", drift_as)):
            as_rows.append(result_row("AS", 0.02, mode, strategy_labels[key], row))
        deltas.append(delta_row("AS", 0.02, mode, classic, drift_as))

        martingale = default["gp_results"][mode]["martingale_gp"]
        drift_gp = default["gp_results"][mode]["hybrid_drift_gp"]
        for key, row in (("martingale_gp", martingale), ("hybrid_drift_gp", drift_gp)):
            gp_default_rows.append(result_row("GP", 5.0, mode, strategy_labels[key], row))
        deltas.append(delta_row("GP γ=5", 5.0, mode, martingale, drift_gp))

        low_martingale = gamma["gp_results"][mode]["martingale_gp"]
        low_drift = gamma["gp_results"][mode]["hybrid_drift_gp"]
        for key, row in (("martingale_gp", low_martingale), ("hybrid_drift_gp", low_drift)):
            gp_gamma_rows.append(result_row("GP", 0.1, mode, strategy_labels[key], row))
        deltas.append(delta_row("GP γ=0.1", 0.1, mode, low_martingale, low_drift))

    as_touch_delta = next(r for r in deltas if r["model"] == "AS" and r["fill_mode"] == "touch")
    as_through_delta = next(r for r in deltas if r["model"] == "AS" and r["fill_mode"] == "through")
    gp_low_through = next(r for r in deltas if r["model"] == "GP γ=0.1" and r["fill_mode"] == "through")
    summary = [{
        "macro_f1_pct": pct(test["macro_f1"]),
        "accuracy_pct": pct(test["accuracy"]),
        "as_touch_delta_hkd": as_touch_delta["net_delta_hkd"],
        "as_through_delta_hkd": as_through_delta["net_delta_hkd"],
        "gp_low_through_delta_hkd": gp_low_through["net_delta_hkd"],
        "test_quotes": signal["test_quotes"],
    }]

    hybrid_source = source(
        "hybrid_result",
        "Hybrid DeepLOB 训练与测试结果",
        "output/results/7709_hybrid_deeplob.json",
        "包含时间切分、标签、训练过程以及 2026-08-07 测试集分类指标。",
    )
    default_source = source(
        "default_integration",
        "AS / GP 默认参数整合回测",
        "0906/output/hybrid_deeplob_as_gp_integration.json",
        "Hybrid drift 与 AS、GP-QVI 在 gamma=5 下的逐日回放机器结果。",
    )
    gamma_source = source(
        "gamma_diagnostic",
        "GP gamma=0.1 机制诊断",
        "0906/output/hybrid_deeplob_as_gp_gamma_0_1_diagnostic.json",
        "仅用于验证 drift 是否会改变离散 GP 动作，不用于测试集选参。",
    )
    report_source = source(
        "report_analysis",
        "报告构建与指标整理",
        "0906/build_hybrid_drift_report.py",
        "从三份机器结果生成图表、差值和技术解释，不重新拟合模型。",
    )
    report_source["query"] = {
        "engine": "duckdb",
        "sql": (
            "SELECT * FROM read_json_auto(["
            "'0906/output/hybrid_deeplob_as_gp_integration.json', "
            "'0906/output/hybrid_deeplob_as_gp_gamma_0_1_diagnostic.json'])"
        ),
        "description": "读取默认整合回测与低 gamma 诊断，并在报告构建脚本中计算 drift 相对基线的差值。",
        "id": "hybrid-drift-delta-inputs",
    }
    model_sources = [
        source("drift_adapter", "Hybrid drift 适配器", "hybrid_drift_adapter.py", "概率到预期价格变动及 drift regime 的转换实现。"),
        source("gp_implementation", "GP drift-QVI 实现", "0906/gp_qvi_model.py", "在时间、drift、spread、库存状态上求解 Bellman/QVI。"),
    ]
    sources = [hybrid_source, default_source, gamma_source, report_source, *model_sources]

    cards = [
        {"id": "macro_f1", "description": "2026-08-07 测试集三分类宏平均 F1。", "dataset": "summary", "sourceId": "hybrid_result", "metrics": [{"label": "Hybrid Macro-F1", "field": "macro_f1_pct", "format": "number", "unit": "%"}]},
        {"id": "as_touch", "description": "乐观 touch 成交代理下，Drift AS 相对 Classic AS 的净收益差。", "dataset": "summary", "sourceId": "default_integration", "metrics": [{"label": "AS touch 增量", "field": "as_touch_delta_hkd", "format": "number", "unit": "HKD"}]},
        {"id": "as_through", "description": "保守 through 成交代理下，Drift AS 相对 Classic AS 的净收益差。", "dataset": "summary", "sourceId": "default_integration", "metrics": [{"label": "AS through 增量", "field": "as_through_delta_hkd", "format": "number", "unit": "HKD"}]},
        {"id": "gp_low", "description": "gamma=0.1 机制诊断，through 下 Drift GP 相对 Martingale GP 的净收益差。", "dataset": "summary", "sourceId": "gamma_diagnostic", "metrics": [{"label": "GP 低 γ through 增量", "field": "gp_low_through_delta_hkd", "format": "number", "unit": "HKD"}]},
    ]

    charts = [
        {
            "id": "drift_distribution", "title": "测试日 drift 状态分布", "subtitle": "3,080 个报价决策；五状态按校准日分位数生成", "type": "bar", "dataset": "drift_states", "sourceId": "default_integration",
            "encodings": {"x": {"field": "state", "type": "nominal", "label": "drift 状态"}, "y": {"field": "share_pct", "type": "quantitative", "label": "决策占比（%）"}}, "yAxisTitle": "%", "valueFormat": "number", "layout": "full",
        },
        {
            "id": "as_net", "title": "AS：drift 改变报价后的净收益", "subtitle": "同一测试日；已扣 2.77 bp 每执行边费用；touch/through 为两种排队代理", "type": "bar", "dataset": "as_results", "sourceId": "default_integration",
            "encodings": {"x": {"field": "fill_mode", "type": "nominal", "label": "成交代理"}, "y": {"field": "net_pnl_hkd", "type": "quantitative", "label": "净收益（HKD）"}, "color": {"field": "strategy", "type": "nominal", "label": "策略"}}, "yAxisTitle": "HKD", "valueFormat": "number", "layout": "full",
        },
        {
            "id": "gp_delta", "title": "GP：加入 drift 后的净收益增量", "subtitle": "gamma=5 为默认结果；gamma=0.1 只用于机制敏感性，不是选参结果", "type": "bar", "dataset": "gp_deltas", "sourceId": "gamma_diagnostic",
            "encodings": {"x": {"field": "scenario", "type": "nominal", "label": "模型 / 成交代理"}, "y": {"field": "net_delta_hkd", "type": "quantitative", "label": "Drift − baseline（HKD）"}}, "yAxisTitle": "HKD", "valueFormat": "number", "layout": "full",
        },
    ]

    tables = [
        {
            "id": "class_metrics", "title": "Hybrid DeepLOB 测试集逐类指标", "subtitle": "最终未参与训练的 2026-08-07；共 61,583 个标签样本", "dataset": "class_metrics", "sourceId": "hybrid_result", "density": "dense", "layout": "full",
            "columns": [{"field": "class", "label": "真实类别", "type": "text"}, {"field": "precision_pct", "label": "Precision %", "format": "number"}, {"field": "recall_pct", "label": "Recall %", "format": "number"}, {"field": "f1_pct", "label": "F1 %", "format": "number"}, {"field": "support", "label": "样本数", "format": "number"}],
        },
        {
            "id": "confusion", "title": "Hybrid DeepLOB 测试集混淆矩阵", "dataset": "confusion", "sourceId": "hybrid_result", "density": "dense", "layout": "full",
            "columns": [{"field": "actual", "label": "真实类别", "type": "text"}, {"field": "pred_down", "label": "预测 down", "format": "number"}, {"field": "pred_stationary", "label": "预测 stationary", "format": "number"}, {"field": "pred_up", "label": "预测 up", "format": "number"}],
        },
        {
            "id": "drift_states", "title": "GP drift regime 明细", "subtitle": "自转移概率已从 4.146 秒观测间隔重标到 3 秒 Bellman 步长", "dataset": "drift_states", "sourceId": "default_integration", "density": "dense", "layout": "full",
            "columns": [{"field": "state", "label": "状态", "type": "text"}, {"field": "drift_hkd_per_second", "label": "HKD/秒", "format": "number"}, {"field": "count", "label": "测试决策数", "format": "number"}, {"field": "share_pct", "label": "占比 %", "format": "number"}, {"field": "self_transition_pct", "label": "3秒自转移 %", "format": "number"}],
        },
        {
            "id": "as_detail", "title": "AS 完整交易结果", "dataset": "as_results", "sourceId": "default_integration", "density": "dense", "layout": "full", "defaultSort": {"field": "fill_mode", "direction": "asc"},
            "columns": [{"field": "fill_mode", "label": "成交代理", "type": "text"}, {"field": "strategy", "label": "策略", "type": "text"}, {"field": "gross_pnl_hkd", "label": "毛收益", "format": "number"}, {"field": "fees_hkd", "label": "费用", "format": "number"}, {"field": "net_pnl_hkd", "label": "净收益", "format": "number", "movement": True}, {"field": "max_drawdown_hkd", "label": "最大回撤", "format": "number"}, {"field": "net_bps", "label": "净bps", "format": "number"}],
        },
        {
            "id": "as_activity", "title": "AS 报价与成交活动", "dataset": "as_results", "sourceId": "default_integration", "density": "dense", "layout": "full",
            "columns": [{"field": "fill_mode", "label": "成交代理", "type": "text"}, {"field": "strategy", "label": "策略", "type": "text"}, {"field": "quoted_bid", "label": "Bid报价", "format": "number"}, {"field": "quoted_ask", "label": "Ask报价", "format": "number"}, {"field": "maker_fills", "label": "Maker成交", "format": "number"}, {"field": "max_inventory_lots", "label": "最大库存", "format": "number"}, {"field": "feeable_turnover_hkd", "label": "计费成交额", "format": "number"}],
        },
        {
            "id": "gp_default", "title": "GP 默认 gamma=5 完整结果", "dataset": "gp_default", "sourceId": "default_integration", "density": "dense", "layout": "full",
            "columns": [{"field": "fill_mode", "label": "成交代理", "type": "text"}, {"field": "strategy", "label": "策略", "type": "text"}, {"field": "net_pnl_hkd", "label": "净收益", "format": "number", "movement": True}, {"field": "maker_fills", "label": "Maker成交", "format": "number"}, {"field": "market_order_lots", "label": "市价手数", "format": "number"}, {"field": "bid_best", "label": "Bid best", "format": "number"}, {"field": "ask_best", "label": "Ask best", "format": "number"}],
        },
        {
            "id": "gp_gamma", "title": "GP gamma=0.1 机制敏感性", "subtitle": "测试日事后诊断，不得作为生产参数选择依据", "dataset": "gp_gamma", "sourceId": "gamma_diagnostic", "density": "dense", "layout": "full",
            "columns": [{"field": "fill_mode", "label": "成交代理", "type": "text"}, {"field": "strategy", "label": "策略", "type": "text"}, {"field": "gross_pnl_hkd", "label": "毛收益", "format": "number"}, {"field": "fees_hkd", "label": "费用", "format": "number"}, {"field": "net_pnl_hkd", "label": "净收益", "format": "number", "movement": True}, {"field": "maker_fills", "label": "成交", "format": "number"}, {"field": "max_drawdown_hkd", "label": "最大回撤", "format": "number"}],
        },
        {
            "id": "gp_actions", "title": "GP 动作计数（默认与低 gamma）", "dataset": "gp_actions", "sourceId": "report_analysis", "density": "dense", "layout": "full",
            "columns": [{"field": "gamma", "label": "gamma", "format": "number"}, {"field": "fill_mode", "label": "成交代理", "type": "text"}, {"field": "strategy", "label": "策略", "type": "text"}, {"field": "bid_best", "label": "Bid best", "format": "number"}, {"field": "bid_improve", "label": "Bid improve", "format": "number"}, {"field": "ask_best", "label": "Ask best", "format": "number"}, {"field": "ask_improve", "label": "Ask improve", "format": "number"}],
        },
        {
            "id": "delta_detail", "title": "Drift 相对基线的结果差", "dataset": "deltas", "sourceId": "report_analysis", "density": "dense", "layout": "full",
            "columns": [{"field": "model", "label": "模型", "type": "text"}, {"field": "fill_mode", "label": "成交代理", "type": "text"}, {"field": "net_delta_hkd", "label": "净收益差", "format": "number", "movement": True}],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
        {"id": "summary", "type": "markdown", "sourceId": "report_analysis", "body": (
            "## 技术结论\n\n"
            "- Drift 已成功进入 AS 和 GP 两个定价框架。\n"
            "- 当前结果不支持稳定提高样本外收益。\n"
            "- Hybrid DeepLOB 测试集宏平均 F1 为 56.24%。\n"
            "- AS 的乐观增量为正 128.79 港元。\n"
            "- AS 的保守增量为负 209.27 港元。\n"
            "- GP 默认参数下，基线与 Drift 动作完全相同。\n"
            "- GP 低风险厌恶诊断会改变动作，但收益不稳健。\n\n"
            "验证等级为有保留地分享。结果适合研究和审阅，不适合上线。"
        )},
        {"id": "signal_quality", "type": "markdown", "sourceId": "hybrid_result", "body": (
            "## 1. Hybrid DeepLOB 信号质量\n\n"
            "训练集覆盖 6 个历史交易日，2026-08-04 用于验证与 checkpoint 选择，2026-08-07 完全留作测试。"
            "测试集 61,583 个样本，Accuracy 55.66%，Balanced Accuracy 56.06%，Macro-F1 56.24%。"
            "模型对 down/up 的精度较高（约 72.0% / 70.2%），但 stationary 精度只有 38.8%；这意味着三分类概率可以提供方向排序，"
            "却不能直接当作价格变动幅度，必须再做概率到 tick move 的校准。"
        )},
        {"id": "class_table", "type": "table", "tableId": "class_metrics", "layout": "full"},
        {"id": "confusion_table", "type": "table", "tableId": "confusion", "layout": "full"},
        {"id": "calibration", "type": "markdown", "sourceId": "default_integration", "body": (
            "## 2. 从分类概率到可交易 drift\n\n"
            "- 校准日使用 3,814 个不重叠报价点。\n"
            "- 目标是未来 20 个快照的中间价变化。\n"
            "- 回归截距为正 0.04616 tick。\n"
            "- 三个概率系数为负 0.15966、负 0.02721、正 0.18686 tick。\n"
            "- 输出截断在正负 2 tick。\n"
            "- 测试日共有 3,080 个报价决策。\n"
            "- 预测均值为正 0.0500 tick，标准差为 0.0800 tick。\n"
            "- 20 个快照的中位实际跨度为 5.6625 秒。\n\n"
            "AS 直接偏移保留价格。GP 先换算为每秒港元漂移，再离散成五状态转移链。"
        )},
        {"id": "drift_chart", "type": "chart", "chartId": "drift_distribution", "layout": "full"},
        {"id": "drift_table", "type": "table", "tableId": "drift_states", "layout": "full"},
        {"id": "as_finding", "type": "markdown", "sourceId": "default_integration", "body": (
            "## 3. AS：偏移确实改变挂单，但收益依赖成交代理\n\n"
            "保守成交代理：\n\n"
            "- 多 155 次被动成交。\n"
            "- 毛收益减少 86.00 港元。\n"
            "- 费用增加 123.27 港元。\n"
            "- 净收益减少 209.27 港元。\n"
            "- 最大回撤增加 207.26 港元。\n\n"
            "乐观成交代理：\n\n"
            "- 多 167 次被动成交。\n"
            "- 毛收益增加 262.00 港元。\n"
            "- 费用增加 133.21 港元。\n"
            "- 净收益增加 128.79 港元。\n"
            "- 最大回撤减少 136.19 港元。\n\n"
            "结论随排队假设翻转，因此尚不能确认存在可交易的超额收益。"
        )},
        {"id": "as_chart", "type": "chart", "chartId": "as_net", "layout": "full"},
        {"id": "as_table", "type": "table", "tableId": "as_detail", "layout": "full"},
        {"id": "as_activity_table", "type": "table", "tableId": "as_activity", "layout": "full"},
        {"id": "gp_default_finding", "type": "markdown", "sourceId": "default_integration", "body": (
            "## 4. GP 默认参数：drift 在价值函数中，但未改变最优动作\n\n"
            "- 风险厌恶参数为 5。\n"
            "- 规划期为 300 秒，共 100 个时间步。\n"
            "- 最大库存为 5 手。\n"
            "- 保守代理下，3,080 次决策全部选择双边不挂单。\n"
            "- 乐观代理下，买卖两侧各有 24 次最优价挂单。\n"
            "- 被动成交 32 次，市价减仓 10 手。\n"
            "- 毛收益 42.00 港元，费用 35.04 港元。\n"
            "- 净收益 6.96 港元。\n"
            "- 基线与 Drift 的动作计数完全相同。\n\n"
            "Drift 已改变状态价值，但幅度不足以跨过离散动作阈值。"
        )},
        {"id": "gp_default_table", "type": "table", "tableId": "gp_default", "layout": "full"},
        {"id": "gp_sensitivity", "type": "markdown", "sourceId": "gamma_diagnostic", "body": (
            "## 5. GP 低 gamma 机制敏感性\n\n"
            "将 gamma 降至 0.1 后，drift 开始改变动作。through 下 Drift GP 相对 baseline 少 4 次成交，净收益从 −HKD 1,046.91 改善至 −HKD 1,043.52；"
            "touch 下多 3 次成交，净收益从 −HKD 247.16 降至 −HKD 250.75。改变量很小，方向仍随成交代理翻转。\n\n"
            "这一组是**机制诊断而非参数选择**：gamma=0.1 是查看测试日后追加的敏感性实验，不能用于宣称样本外提升。"
            "它只能证明 GP drift 状态确实能穿透 Bellman 价值函数并改变报价动作。"
        )},
        {"id": "gp_delta_chart", "type": "chart", "chartId": "gp_delta", "layout": "full"},
        {"id": "gp_gamma_table", "type": "table", "tableId": "gp_gamma", "layout": "full"},
        {"id": "delta_table", "type": "table", "tableId": "delta_detail", "layout": "full"},
        {"id": "scope", "type": "markdown", "sourceId": "report_analysis", "body": (
            "## 6. 范围、单位与收益定义\n\n"
            "标的是 HK 07709。信号校准日为 2026-08-04，最终回放日为 2026-08-07。神经网络输入价格与执行回放价格之间存在 100 倍尺度差，"
            "整合器已显式转换；测试日合法 tick 为 HKD 0.02。每手 100 股。\n\n"
            "净收益 = 报价和日终平仓现金流形成的毛收益 − 每个执行边 2.77 bp 的全包费用。"
            "`touch` 表示未来窗口内触及挂单价即可成交，较乐观；`through` 要求价格穿过挂单价，较保守。两者是排队敏感性边界，不是统计置信区间。"
        )},
        {"id": "methodology", "type": "markdown", "sourceId": "gp_implementation", "body": (
            "## 7. 方法与实现\n\n"
            "AS 版本以经典库存 reservation-price 逻辑为基线，加入 `signal_strength × expected_move_ticks`，随后按风险厌恶和订单到达衰减确定双边距离。"
            "参数为风险厌恶 0.02/tick、订单衰减 1.0/tick、20-snapshot 方差 5.39785 tick²、最大库存 5 手。\n\n"
            "GP 版本使用 `(剩余时间, drift状态, spread状态, inventory)` 状态，在每个 Bellman 步加入 `q × 100 × μ × dt` 持仓收益，"
            "并用校准得到的五状态 Markov 转移传播 drift。零 drift 时单元测试要求并确认精确回退到原 martingale GP。"
            "执行回放使用 5ms 生效延迟及未来 20 snapshot 成交窗口。"
        )},
        {"id": "validation", "type": "markdown", "sourceId": "report_analysis", "body": (
            "## 8. 验证与稳健性\n\n"
            "- 19 个相关单元测试全部通过，包括零 drift 等价、状态重标、动作求解和适配器输入输出。\n"
            "- 报告从机器 JSON 直接派生；毛收益减费用与净收益、买卖成交合计与 maker 成交数均已交叉核对。\n"
            "- 最关键的稳健性检查是 touch/through：AS 和低 gamma GP 的增量方向都发生翻转，因此不能把单一成交代理的正值解释为稳健提升。\n"
            "- 默认 GP 的 baseline/drift 动作逐项相同，低 gamma 下动作计数不同，排除了‘代码路径根本未使用 drift’这一解释。"
        )},
        {"id": "limitations", "type": "markdown", "body": (
            "## 9. 限制与解释边界\n\n"
            "- 最终测试只有一个交易日。\n"
            "- 尚无日间方差或置信区间。\n"
            "- 成交代理不含真实排队位置。\n"
            "- 未建模隐藏流动性和盘口冲击。\n"
            "- 未建模最低佣金和借券限制。\n"
            "- AS 参数尚未严格滚动选择。\n"
            "- 验证集同时用于模型选择和概率校准。\n"
            "- GP 仅有三个离散限价动作。\n"
            "- 漂移与价差转移按条件独立组合。"
        )},
        {"id": "next_steps", "type": "markdown", "body": (
            "## 10. 下一步建议\n\n"
            "1. 在多个未见交易日做滚动前推验证。\n"
            "2. 每天只使用过去数据重新校准。\n"
            "3. 将 GP 扩展到多档报价距离。\n"
            "4. 记录各动作之间的价值差。\n"
            "5. 用真实排队位置重放成交。\n"
            "6. 加入最低佣金、撤单和冲击成本。\n"
            "7. 只在滚动验证集选择参数。\n"
            "8. 最终测试集保留一次性评估。\n"
            "9. 跨日且跨成交代理稳定后，再进入模拟交易。"
        )},
        {"id": "questions", "type": "markdown", "body": (
            "## 11. 仍需回答的问题\n\n"
            "DeepLOB 的有效 horizon 应固定为 20 snapshot，还是按市场事件速度动态换算？GP 是否需要把 drift 与 spread 合并为联合状态链？"
            "真实账户费用和 queue position 数据能否获得？这三项会直接决定信号是否足以跨过 GP 动作阈值，以及 AS 的 touch 优势是否可成交。"
        )},
    ]

    # Keep the technical narrative on one readable report column. Without an
    # explicit full-width layout, adjacent markdown blocks may be placed in the
    # two-column grid and long mixed Chinese/English terms can widen the page.
    for block in blocks:
        if block["type"] in {"markdown", "metric-strip"}:
            block.setdefault("layout", "full")
        if block["type"] == "markdown":
            # Evidence is attached to the adjacent native table/chart. The
            # reader's inline markdown-source tooltip does not wrap long local
            # JSON paths reliably and can widen the document viewport.
            block.pop("sourceId", None)
            # The current portable reader has a known width regression for
            # long ordered/unordered markdown lists. Preserve each item as a
            # short paragraph so the same information remains visible.
            block["body"] = re.sub(r"\n(?:-|\d+\.)\s+", "\n\n", block["body"])
            if "\n\n" in block["body"]:
                heading, narrative = block["body"].split("\n\n", 1)
                block["body"] = heading + "\n\n" + re.sub(r"\s+", " ", narrative).strip()
    # The portable reader already renders manifest.title in its page header.
    # A second long H1 block can exceed the reader's content grid at desktop
    # widths, so keep one canonical title only.
    blocks = [block for block in blocks if block["id"] != "title"]

    manifest = {
        "version": 1, "surface": "report", "title": TITLE,
        "description": "Hybrid DeepLOB drift 与 AS、GP-QVI 的单日样本外整合、机制诊断和成交敏感性结果。",
        "generatedAt": generated_at, "cards": cards, "charts": charts, "tables": tables,
        "sources": sources, "blocks": blocks,
    }
    artifact = {
        "surface": "report", "manifest": manifest,
        "snapshot": {
            "version": 1, "generatedAt": generated_at, "status": "ready",
            "datasets": {
                "summary": summary,
                "class_metrics": class_metrics,
                "confusion": confusion,
                "drift_states": drift_states,
                "as_results": as_rows,
                "gp_default": gp_default_rows,
                "gp_gamma": gp_gamma_rows,
                "gp_actions": gp_default_rows + gp_gamma_rows,
                "gp_deltas": [row for row in deltas if row["model"].startswith("GP")],
                "deltas": deltas,
            },
        },
        "sources": sources,
        "package_info": {},
    }
    ARTIFACT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(ARTIFACT)


if __name__ == "__main__":
    main()
