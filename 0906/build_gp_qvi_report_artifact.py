#!/usr/bin/env python3
"""Build the canonical Data Analytics artifact for the GP/QVI backtest report."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
RESULTS = OUTPUT / "july_2026_gp_qvi_new_fill_results.json"
DAILY = OUTPUT / "july_2026_gp_qvi_new_fill_daily.csv"
ARTIFACT = OUTPUT / "gp_qvi_report_artifact.json"
TITLE = "7709 的 GP-QVI 一个月回测"


def source(source_id: str, label: str, path: str, description: str) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "description": description,
    }


def main() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    daily_rows = list(csv.DictReader(DAILY.open(encoding="utf-8-sig")))
    aggregate = results["aggregate"]

    summary = aggregate["through"]["gp_qvi"]
    womo = aggregate["through"]["gp_qvi_womo"]
    touch = aggregate["touch"]["gp_qvi"]
    summary_rows = [{
        "through_net": summary["net_pnl_hkd"],
        "through_gross": summary["gross_pnl_hkd"],
        "through_fees": summary["all_in_fees_hkd"],
        "through_womo_net": womo["net_pnl_hkd"],
        "touch_net": touch["net_pnl_hkd"],
        "test_days": summary["test_days"],
        "maker_fills": summary["maker_fills"],
    }]
    headline_pnl = [
        {"strategy": "GP-QVI", "net_pnl_hkd": summary["net_pnl_hkd"]},
        {"strategy": "GP WoMO", "net_pnl_hkd": womo["net_pnl_hkd"]},
    ]
    monthly_comparison = []
    strategy_labels = {
        "gp_qvi": "GP-QVI",
        "gp_qvi_womo": "GP WoMO",
        "constant_best": "恒定最优价",
    }
    for mode in ("through", "touch"):
        for strategy in ("gp_qvi", "gp_qvi_womo"):
            row = aggregate[mode][strategy]
            monthly_comparison.append({
                "fill_mode": mode,
                "strategy": strategy_labels[strategy],
                "net_pnl_hkd": row["net_pnl_hkd"],
                "gross_pnl_hkd": row["gross_pnl_hkd"],
                "fees_hkd": row["all_in_fees_hkd"],
                "maker_fills": row["maker_fills"],
                "market_order_lots": row["market_order_lots"],
            })
    daily_through = []
    for row in daily_rows:
        if row["fill_mode"] != "through":
            continue
        daily_through.extend([
            {
                "date": row["date"],
                "strategy": "GP-QVI",
                "net_pnl_hkd": float(row["gp_qvi_net_pnl_hkd"]),
            },
            {
                "date": row["date"],
                "strategy": "GP WoMO",
                "net_pnl_hkd": float(row["gp_qvi_womo_net_pnl_hkd"]),
            },
        ])
    aggregate_table = []
    for mode in ("through", "touch"):
        for strategy in ("gp_qvi", "gp_qvi_womo", "constant_best"):
            row = aggregate[mode][strategy]
            aggregate_table.append({
                "fill_mode": mode,
                "strategy": strategy_labels[strategy],
                "gross_pnl_hkd": row["gross_pnl_hkd"],
                "fees_hkd": row["all_in_fees_hkd"],
                "net_pnl_hkd": row["net_pnl_hkd"],
                "profitable_days": row["profitable_days"],
                "maker_fills": row["maker_fills"],
                "market_order_lots": row["market_order_lots"],
                "max_drawdown_hkd": row["day_close_max_drawdown_hkd"],
            })

    results_source = source(
        "gp_results",
        "GP-QVI 月度回测机器结果",
        "0906/output/july_2026_gp_qvi_new_fill_results.json",
        "从 19 个样本外交易日逐日结果汇总月度收益、成交、费用与回撤。",
    )
    results_source["query"] = {
        "engine": "duckdb",
        "language": "sql",
        "sql": (
            "SELECT * FROM read_json_auto("
            "'0906/output/july_2026_gp_qvi_new_fill_results.json')"
        ),
        "description": "读取经程序校验的 GP-QVI 月度机器结果；报告构建脚本再展开 aggregate 与 tests 字段。",
        "tables_used": ["0906/output/july_2026_gp_qvi_new_fill_results.json"],
        "filters": ["instrument = HK 07709", "period = 2026-07", "19 full-day out-of-sample tests"],
        "metric_definitions": {
            "net_pnl_hkd": "gross cash trading PnL minus all-in execution fees",
            "all_in_fees_hkd": "feeable turnover multiplied by 2.77 basis points per execution side",
        },
        "executed_at": "2026-09-10T09:00:00+08:00",
    }
    model_source = source(
        "gp_model",
        "GP-QVI 求解与回放实现",
        "0906/gp_qvi_model.py",
        "实现 GP 的状态、控制、滚动估计、Bellman/QVI 离散化及 0906 新成交逻辑。",
    )
    paper_source = {
        "id": "gp_paper",
        "label": "Guilbaud-Pham 原论文",
        "path": "0906/papers/Guilbaud_Pham_2011_Optimal_High_Frequency_Trading_with_Limit_and_Market_Orders.pdf",
    }

    manifest = {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": "HK 07709，2026 年 7 月，论文形式 GP-QVI 与 0906 新成交逻辑。",
        "generatedAt": "2026-09-10T09:00:00+08:00",
        "cards": [
            {
                "id": "through_net",
                "description": "保守 through 口径、2.77 bp 每执行边费用后的整月收益。",
                "dataset": "summary",
                "sourceId": "gp_results",
                "metrics": [{"label": "GP-QVI 月度净收益", "field": "through_net", "format": "number", "unit": "HKD"}],
            },
            {
                "id": "through_gross",
                "description": "扣费前现金交易损益。",
                "dataset": "summary",
                "sourceId": "gp_results",
                "metrics": [{"label": "through 毛收益", "field": "through_gross", "format": "number", "unit": "HKD"}],
            },
            {
                "id": "through_fees",
                "description": "官方费用 1.27 bp 加券商佣金假设 1.50 bp。",
                "dataset": "summary",
                "sourceId": "gp_results",
                "metrics": [{"label": "through 全包费用", "field": "through_fees", "format": "number", "unit": "HKD"}],
            },
            {
                "id": "through_womo",
                "description": "关闭盘中市价单，只用限价单管理库存。",
                "dataset": "summary",
                "sourceId": "gp_results",
                "metrics": [{"label": "WoMO 月度净收益", "field": "through_womo_net", "format": "number", "unit": "HKD"}],
            },
        ],
        "charts": [
            {
                "id": "headline_pnl",
                "title": "through 口径月度净收益",
                "subtitle": "19 个样本外交易日；HKD；已扣除 2.77 bp 每执行边费用",
                "type": "bar",
                "dataset": "headline_pnl",
                "sourceId": "gp_results",
                "encodings": {
                    "x": {"field": "strategy", "type": "nominal", "label": "策略"},
                    "y": {"field": "net_pnl_hkd", "type": "quantitative", "label": "月度净收益", "format": "number"},
                },
                "yAxisTitle": "HKD",
                "valueFormat": "number",
                "layout": "full",
            },
            {
                "id": "daily_through_pnl",
                "title": "through 口径逐日净收益",
                "subtitle": "GP-QVI 损失集中于少数日期；WoMO 大多数日期选择不交易",
                "type": "line",
                "dataset": "daily_through",
                "sourceId": "gp_results",
                "encodings": {
                    "x": {"field": "date", "type": "temporal", "label": "交易日"},
                    "y": {"field": "net_pnl_hkd", "type": "quantitative", "label": "净收益", "format": "number"},
                    "color": {"field": "strategy", "type": "nominal", "label": "策略"},
                },
                "yAxisTitle": "HKD",
                "valueFormat": "number",
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "aggregate_detail",
                "title": "策略与成交口径汇总",
                "subtitle": "2026 年 7 月 3 日至 30 日，共 19 个样本外完整交易日",
                "dataset": "aggregate_detail",
                "sourceId": "gp_results",
                "defaultSort": {"field": "net_pnl_hkd", "direction": "desc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "fill_mode", "label": "成交口径", "type": "text"},
                    {"field": "strategy", "label": "策略", "type": "text"},
                    {"field": "gross_pnl_hkd", "label": "毛收益", "format": "number"},
                    {"field": "fees_hkd", "label": "费用", "format": "number"},
                    {"field": "net_pnl_hkd", "label": "净收益", "format": "number", "movement": True},
                    {"field": "maker_fills", "label": "Maker 成交", "format": "number"},
                    {"field": "market_order_lots", "label": "市价手数", "format": "number"},
                ],
            },
        ],
        "sources": [results_source, model_source, paper_source],
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "sourceId": "gp_results",
                "body": (
                    "## 技术结论\n\n"
                    "**按保守的 `through` 成交口径，论文形式 GP-QVI 在 19 个样本外交易日的全包净收益为 "
                    "HKD −236.94。** 毛收益 HKD 468.00，不足以覆盖 HKD 704.94 的执行费用。"
                    "乐观 `touch` 口径净收益为 HKD −3,830.95。\n\n"
                    "关闭盘中市价单的 WoMO 版本反而分别实现 HKD +58.48（through）和 HKD +202.60（touch）。"
                    "这说明 GP 的库存控制机制已经复现，但在 7709 的费用与新成交代理下，频繁用市价单归零库存并不经济。"
                ),
            },
            {"id": "metrics", "type": "metric-strip", "cardIds": ["through_net", "through_gross", "through_fees", "through_womo"]},
            {
                "id": "finding_market_orders",
                "type": "markdown",
                "sourceId": "gp_results",
                "body": (
                    "## 市价减仓没有赚回执行成本\n\n"
                    "GP-QVI 在 through 下完成 231 笔 maker 成交，并用 95 手市价单减仓；相对 WoMO，它多获得 "
                    "HKD 378.00 毛收益，却多付 HKD 673.42 费用，净收益因此低 HKD 295.42。touch 下差距更大："
                    "多出的毛收益只有 HKD 154.98，但额外费用达到 HKD 4,188.53。\n\n"
                    "因此这里不是库存惩罚失效，而是惩罚确实促使策略快速回到零库存；问题在于这种风险交换在当前费率下太贵。"
                ),
            },
            {"id": "headline_chart", "type": "chart", "chartId": "headline_pnl", "layout": "full"},
            {
                "id": "finding_daily",
                "type": "markdown",
                "sourceId": "gp_results",
                "body": (
                    "## through 损失集中在少数交易日\n\n"
                    "GP-QVI 有 8 个盈利日、10 个亏损日和 1 个零交易收益日；最大单日收益为 HKD +95.41（7 月 16 日），"
                    "最差为 HKD −183.74（7 月 21 日）。WoMO 有 17 天选择不产生净交易收益，体现了高惩罚和费用下的保守最优解。"
                ),
            },
            {"id": "daily_chart", "type": "chart", "chartId": "daily_through_pnl", "layout": "full"},
            {
                "id": "scope",
                "type": "markdown",
                "sourceId": "gp_results",
                "body": (
                    "## 范围与收益定义\n\n"
                    "样本为 HK 07709 的 2026 年 7 月数据。7 月 2 日只用于初始校准，之后 19 个完整交易日计入样本外 PnL；"
                    "每天仅使用此前最多 5 个完整交易日估计参数。净收益定义为实际报价现金流和日终平仓现金流之和，减去每个执行边 "
                    "2.77 bp（官方 1.27 bp + 券商佣金假设 1.50 bp）。未给定账户资本，因此不报告资本收益率。"
                ),
            },
            {
                "id": "method",
                "type": "markdown",
                "sourceId": "gp_model",
                "body": (
                    "## 论文模型如何落到 7709\n\n"
                    "实现采用论文的线性效用降维 `v=x+yp+φ(t,y,s)`，在 `(剩余时间, 库存, spread)` 上反向求解。"
                    "每侧控制为不挂、当前最优价或改善一个合法 tick；GP-QVI 还可用市价单减少库存，WoMO 禁用该冲击控制。"
                    "库存范围为 ±10 手，每手 100 股；`T=300s`、100 个时间步、`γ=5`。\n\n"
                    "spread 转移矩阵、六个日内时段的 Poisson 时钟强度和按 spread/方向/报价层级区分的成交强度，全部由滚动历史估计。"
                    "回放沿用新成交逻辑：5ms 后生效，只检查 `t+1…t+20`，未来成交价或对手方 L1 报价任一先满足即成交。"
                ),
            },
            {"id": "detail_table", "type": "table", "tableId": "aggregate_detail", "layout": "full"},
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## 限制、稳健性与解释边界\n\n"
                    "- 这是基于逐笔成交和 L1 触价的可执行性代理，不含 queue-ahead、隐藏流动性、交易所确认、市场冲击与借券成本。\n"
                    "- `touch` 与 `through` 分别代表乐观和保守的排队敏感性，不是置信区间；两者分别校准政策，不能当成只改变一个条件的严格配对实验。\n"
                    "- 论文中的 EUR/股参数不能无量纲地直接搬到 HKD/手。本复现保留 `γ=5` 的结构，将库存单位改为 100 股一手，因此结果应视为论文形式的工程复现，而非原表参数的跨市场同量纲复制。\n"
                    "- 月初合法 tick 为 HKD 0.05，之后主要为 HKD 0.02；实现按当日盘口网格识别，避免固定 tick 造成非法改善报价。\n"
                    "- 仅按官方 1.27 bp 计算时，through 的 GP-QVI 为 HKD +144.80；加入 1.50 bp 佣金假设后转为 HKD −236.94，结论对费用高度敏感。"
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## 建议的下一步\n\n"
                    "1. 把市价冲击控制加入费用阈值：只有预期库存风险下降超过跨 spread 和佣金时才减仓。\n"
                    "2. 对 `γ` 做严格 walk-forward 灵敏度分析，同时保留论文的 WoMO 作为低换手基线。\n"
                    "3. 若目标是可交易收益，优先研究 WoMO；当前证据不支持直接上线带市价归零的 GP-QVI。"
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## 仍需回答的问题\n\n"
                    "真实券商是否有最低佣金、maker rebate 或分层费率？可获得的 queue-ahead 数据是否足以替代触价代理？"
                    "这两项会最直接地改变市价减仓与限价等待之间的最优边界。"
                ),
            },
        ],
    }
    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": "2026-09-10T09:00:00+08:00",
            "status": "ready",
            "datasets": {
                "summary": summary_rows,
                "headline_pnl": headline_pnl,
                "monthly_comparison": monthly_comparison,
                "daily_through": daily_through,
                "aggregate_detail": aggregate_table,
            },
        },
        "sources": [results_source, model_source, paper_source],
        "package_info": {},
    }
    ARTIFACT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(ARTIFACT)


if __name__ == "__main__":
    main()
