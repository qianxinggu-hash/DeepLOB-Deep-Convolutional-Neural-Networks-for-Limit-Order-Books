#!/usr/bin/env python3
"""Technical report for the 0920 signed-inventory GP gamma rerun."""
from __future__ import annotations
import json
import os
import sqlite3
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
TITLE = "GP-QVI Signed-Inventory Gamma Rerun"
ANCHORS = (5, 1, 0.5, 0.1, 0.05, 0.01, 0)


def main() -> None:
    result = json.loads((OUT / "gamma_results.json").read_text())
    assert result["validation"]["all_passed"]
    variants = result["variants"]
    base = variants["gamma_5"]["aggregate"]["through"]
    point = variants["gamma_0.1"]["aggregate"]["through"]
    rows = []
    for key, v in variants.items():
        gamma = v["config"]["inventory_penalty_gamma"]
        for mode, a in v["aggregate"].items():
            rows.append({
                "gamma": gamma, "gamma_label": f"{gamma:.6g}", "mode": mode,
                "fills": a["maker_fills"], "fills_per_day": round(a["maker_fills_per_day"], 2),
                "quoted_share_pct": round(a["quoted_side_share"] * 100, 3),
                "fill_rate_pct": round(a["maker_fill_rate"] * 100, 2),
                "gross": round(a["gross_pnl_hkd"], 2), "fees": round(a["all_in_fees_hkd"], 2),
                "net": round(a["net_pnl_hkd"], 2), "mean_daily_net": round(a["mean_daily_net_pnl_hkd"], 2),
                "daily_net_std": round(a["daily_net_pnl_std_hkd"], 2),
                "peak_inventory": a["max_abs_inventory_lots"],
                "min_inventory": a["min_inventory_lots"],
                "max_inventory": a["max_inventory_lots"],
                "time_mean_abs_inventory": round(a["time_mean_abs_inventory_lots"], 4),
                "inventory_rms": round(a["time_rms_inventory_lots"], 4),
                "nonzero_inventory_time_pct": round(a["nonzero_inventory_time_share"] * 100, 2),
                "negative_inventory_time_pct": round(a["negative_inventory_time_share"] * 100, 2),
                "positive_inventory_time_pct": round(a["positive_inventory_time_share"] * 100, 2),
                "short_inventory_lot_seconds": round(a["short_inventory_lot_seconds"], 3),
                "long_inventory_lot_seconds": round(a["long_inventory_lot_seconds"], 3),
                "q2_seconds": round(a["squared_inventory_lot2_seconds"], 3),
                "penalty_hkd": round(a["realized_inventory_penalty_hkd"], 2),
                "policy_market_orders": a["policy_market_orders"],
                "policy_market_lots": a["policy_market_order_lots"],
                "terminal_lots": a["terminal_flatten_lots"],
                "impulse_state_pct": round(a["mean_full_horizon_impulse_state_share"] * 100, 2),
                "worst_intraday_drawdown": round(a["worst_intraday_drawdown_hkd"], 2),
                "close_drawdown": round(a["day_close_max_drawdown_hkd"], 2),
                "active_days": a["active_days"],
                "one_lot_penalty_per_3s": 3 * gamma,
            })
    through = [r for r in rows if r["mode"] == "through"]
    touch = [r for r in rows if r["mode"] == "touch"]
    anchors = [r for r in through if r["gamma"] in ANCHORS]
    datasets = {"through": through, "touch": touch, "anchors": anchors}
    sql = "SELECT dataset, row_json FROM report_datasets ORDER BY dataset, row_index"
    database = OUT / "report_datasets.sqlite"
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    with connection:
        connection.execute("CREATE TABLE report_datasets (dataset TEXT, row_index INTEGER, row_json TEXT)")
        connection.executemany("INSERT INTO report_datasets VALUES (?, ?, ?)", [
            (name, i, json.dumps(row, ensure_ascii=False)) for name, values in datasets.items()
            for i, row in enumerate(values)])
    datasets = {}
    for name, row_json in connection.execute(sql):
        datasets.setdefault(name, []).append(json.loads(row_json))
    connection.close()
    (OUT / "report_data_query.sql").write_text(sql + ";\n")

    sources = [
        {"id": "replay", "label": "0920 GP-QVI 有符号库存重跑", "path": "0920/output/report_datasets.sqlite",
         "description": "Python展开已核对的重跑结果后，通过实际执行的SQLite查询读取报告数据。",
         "query": {"engine": "sqlite", "language": "sql", "sql": sql,
                   "tables_used": ["report_datasets", "0920/output/report_datasets.sqlite",
                                   "0920/output/gamma_results.json"],
                   "filters": ["HK 07709", "2026年7月19个完整测试交易日", "原GP-QVI限价+市价减仓控制"],
                   "metric_definitions": {
                       "fills": "单侧100股限价单成交笔数，不是往返交易数；不包含市价单",
                       "net": "含日终平仓的现金毛损益减每执行边2.77bp费用",
                       "inventory_rms": "sqrt(全日库存平方时间积分/全日回放时长)，含持仓跨午休",
                       "negative_inventory_time_pct": "库存q<0的墙钟时间/全日回放墙钟时间；允许卖出开空",
                       "short_inventory_lot_seconds": "Σ max(-q,0)×持仓墙钟秒数",
                       "q2_seconds": "Σ每段持仓手数²×持仓墙钟秒数",
                       "penalty_hkd": "gamma×全日库存平方时间积分；诊断量，不是300秒Bellman价值",
                       "impulse_state_pct": "在300秒剩余期限的库存×spread状态中，市价目标不同于当前库存的比例；19天和6时段等权平均，不是实际访问占比",
                       "daily_net_std": "19个每日净损益的样本标准差，不是Monte Carlo路径风险",
                   }, "executed_at": result["as_of"]}},
        {"id": "paper", "label": "Guilbaud–Pham 原论文：3.3节与第4节",
         "path": result["provenance"]["paper"],
         "description": "已核对PDF第17页表4和第21页表6；目标函数与均值库存惩罚推导见3.1–3.3节。"},
        {"id": "model", "label": "0906 GP-QVI 模型及库存单位",
         "path": "0906/gp_qvi_model.py",
         "description": "原求解器的库存网格为-10至+10手；100股一手，双侧限价允许从零卖出形成空头，市价单只向零减仓。"},
        {"id": "baseline", "label": "0913 gamma研究基线", "path": "0913/output/gamma_results.json",
         "description": "用于逐日期、逐gamma、逐成交模型核对成交、库存峰值和现金损益。"},
    ]
    charts = []

    def chart(id_, title, type_, dataset, x, y, x_label, y_label, subtitle):
        charts.append({"id": id_, "title": title, "subtitle": subtitle,
                       "type": type_, "dataset": dataset, "sourceId": "replay",
                       "encodings": {"x": {"field": x, "type": "nominal" if x == "gamma_label" else "quantitative", "label": x_label},
                                     "y": {"field": y, "type": "quantitative", "label": y_label}}})

    chart("fills", "Gamma 与 Maker 成交笔数", "bar", "anchors", "gamma_label", "fills",
          "gamma", "单侧成交笔数", "七个核对档位；19天；through；原限价+市价控制不变")
    chart("inventory", "Gamma 与时间加权库存", "line", "anchors", "gamma_label", "inventory_rms",
          "gamma", "库存RMS（手）", "持仓平方时间积分计算；包括跨午休持仓；不是每日峰值平均")
    chart("short_share", "Gamma 与负库存持有时间占比", "bar", "through", "gamma_label", "negative_inventory_time_pct",
          "gamma", "负库存时间占比（%）", "21档gamma；库存q<0的墙钟时间占全日回放时间；through")
    chart("impulse", "Gamma 与理论市价减仓区域", "bar", "anchors", "gamma_label", "impulse_state_pct",
          "gamma", "状态占比（%）", "300秒剩余期限、库存×spread网格；各日和时段等权，不是实际访问率")
    chart("frontier", "每日净损益的经验风险与收益", "scatter", "through", "daily_net_std", "mean_daily_net",
          "每日净损益标准差（HKD）", "平均每日净损益（HKD）",
          "21档gamma、每档同样19天；经验点，不是论文Monte Carlo有效前沿或置信区间")
    tables = []

    def table(id_, title, dataset, cols):
        tables.append({"id": id_, "title": title, "dataset": dataset, "sourceId": "replay",
                       "density": "spacious", "layout": "full",
                       "columns": [{"field": field, "label": label,
                                    **({"type": "text"} if field == "gamma_label" else {"format": "number"})}
                                   for field, label in cols]})

    table("economics", "21档 gamma 的交易与费用", "through", [
        ("gamma_label", "gamma"), ("fills", "Maker笔数"), ("gross", "毛收益HKD"),
        ("fees", "费用HKD"), ("net", "净收益HKD"), ("active_days", "成交天数")])
    table("inventory", "库存暴露与市价调整明细", "through", [
        ("gamma_label", "gamma"), ("min_inventory", "最小库存手"), ("max_inventory", "最大库存手"),
        ("inventory_rms", "库存RMS手"), ("negative_inventory_time_pct", "负库存时间%"),
        ("positive_inventory_time_pct", "正库存时间%"),
        ("policy_market_orders", "盘中市价次数"), ("impulse_state_pct", "理论市价区域%")])
    table("touch", "Touch 执行模型敏感性", "touch", [
        ("gamma_label", "gamma"), ("fills", "Maker笔数"), ("net", "净收益HKD"),
        ("inventory_rms", "库存RMS手"), ("policy_market_orders", "盘中市价次数")])
    blocks = [{"id": "title", "type": "markdown", "body": f"# {TITLE}"}]

    def prose(id_, title, body, source="replay"):
        block = {"id": id_, "type": "markdown", "body": f"## {title}\n\n{body}"}
        if source:
            block["sourceId"] = source
        blocks.append(block)

    def visual(kind, id_):
        blocks.append({"id": f"{kind}_{id_}", "type": kind, f"{kind}Id": id_, "layout": "full"})

    prose("summary", "Technical Summary", (
        "- **0920显式采用-10至+10手的有符号库存域。** 双侧限价单可使库存从0变为负数，市价单仍只向0减仓；21档gamma、19个测试日和两种成交模型均观察到负库存。\n"
        "- **重跑没有改变0913的交易决策或现金流。** 全部21档gamma均通过逐日成交、库存峰值、毛损益、费用和净损益一致性核对；新增内容是空头暴露指标与显式验证。\n"
        "- **只研究原GP-QVI方法中的gamma。** 每档均保留限价单和市价库存调整，固定其余参数；本轮比较21档gamma。\n"
        f"- **降低gamma能增加挂单与成交，但当前真实数据回放中的净收益恶化。** gamma从5降至0.1，"
        f"through成交{base['maker_fills']:,}→{point['maker_fills']:,}笔，净收益"
        f"HKD {base['net_pnl_hkd']:+,.2f}→{point['net_pnl_hkd']:+,.2f}。\n"
        f"- **库存风险用持仓时间衡量。** 同两档库存RMS为{base['time_rms_inventory_lots']:.4f}→"
        f"{point['time_rms_inventory_lots']:.4f}手；理论市价控制区域和实际减仓次数分别报告。\n"
        "- **不按整月收益挑选“最优gamma”。** 本次提供收益、库存和政策区域的敏感性证据；参数选择须使用测试日之前的数据。"))
    prose("scope", "样本、库存域与指标定义", (
        "样本为HK 07709的2026年7月，7月2日只用于初始校准，随后19个完整交易日纳入结果。"
        "maker笔数按单侧100股限价单成交计数，市价减仓另外统计。净收益包含日终强制平仓以及每个执行边2.77bp费用。"
        "每天使用此前最多五个完整交易日估计spread转移、日内时钟和成交强度；本轮仅改变gamma，"
        "固定-10至+10手库存域、300秒规划期限、100步以及5ms/t+1…t+20执行规则。"
        "负库存时间占比以q<0的墙钟秒数除以全日回放墙钟秒数；空头手秒为Σmax(-q,0)×秒。"))

    prose("paper_method", "论文中的gamma控制的是收益与持仓风险的权衡", (
        "本研究对应论文3.3节的均值效用与库存惩罚方法：U(x)=x、g(y)=y²，"
        "目标为期末清算财富的期望减去gamma乘持仓平方的时间积分。"
        "限价单用于赚取价差，市价单用于立即调整库存，二者共同进入QVI。"
        "论文第4节用不同gamma构建风险收益前沿，并指出增大gamma会扩大市价交易区域、缩小容忍的库存范围。"
        "表4列出gamma=5；新网格按表6的等比减半结构生成50/2^i（i=0…13），并加入此前7个核对档位。"
        "表6的数值来自原模型Monte Carlo实验，不能当成7709上的最优参数。", "paper"))
    prose("units", "先统一库存单位，再解释gamma的大小", (
        "当前代码库存q以100股一手计，运行成本为gamma_lot×q²×秒，gamma_lot单位为HKD/手²/秒。"
        "若改用股数Y=100q，同一个惩罚项对应gamma_share=gamma_lot/10,000。"
        "因此保留数字5并不等于与论文的股数库存同量纲复制。"
        "在固定3秒一步时，持有1手的单步惩罚为3gamma港元；gamma=5时是15港元，gamma=0.1时是0.3港元。"
        "本轮研究的是0906工程映射下的gamma，未擅自改变币种、库存单位或费用。"
        "gamma=0只取消本方法中的运行惩罚，仍使用线性效用，不会自动变成指数效用方法。", "model"))
    prose("activity", "成交增加首先来自更愿意挂单", (
        f"gamma=5时，{base['quote_decisions']:,}次报价决策仅挂出{base['quoted_bid'] + base['quoted_ask']:,}侧订单，"
        f"挂单侧占比为{base['quoted_side_share']:.3%}；已挂订单成交率为{base['maker_fill_rate']:.1%}。"
        f"gamma=0.1时挂单侧占比为{point['quoted_side_share']:.2%}，成交{point['maker_fills']:,}笔。"
        "图保留七个原核对档位，完整21档在明细表中；成交条件和成交强度的校准口径没有随gamma改变。"))
    visual("chart", "fills")
    prose("economics", "交易量增加需要和扣费收益一起判断", (
        f"gamma=0.1的毛收益为HKD {point['gross_pnl_hkd']:+,.2f}、费用为HKD {point['all_in_fees_hkd']:,.2f}，"
        "因此亏损同时来自交易现金流和费用。下表将两部分分别列出。"
        "gamma从5增至50的实际成交与现金损益相同，显示回放路径上结果已饱和；"
        "这不等于整个理论状态网格的控制区域完全相同。"))
    visual("table", "economics")
    prose("inventory_risk", "库存平方时间积分把仓位大小与持仓时间结合起来", (
        "库存RMS=sqrt(持仓手数²的时间积分/回放时长)，而峰值库存只记录最大瞬时仓位。"
        f"gamma=5的全期平方库存暴露为{base['squared_inventory_lot2_seconds']:,.3f}手²秒，"
        f"gamma=0.1为{point['squared_inventory_lot2_seconds']:,.3f}手²秒。"
        "图显示gamma对时间加权库存的影响，不要求逐档严格单调。"
        "计时从首报价+5ms到日终到期+5ms，持仓跨午休时也计入风险；全日gamma乘该积分仅作诊断，"
        "不等于每次滚动300秒求解的Bellman价值。"))
    visual("chart", "inventory")
    prose("signed_inventory", "负库存不是边界异常，而是策略实际访问的状态", (
        f"gamma=5在负库存中的时间占比为{base['negative_inventory_time_share']:.2%}，"
        f"正库存时间占比为{base['positive_inventory_time_share']:.2%}；"
        f"gamma=0.1对应{point['negative_inventory_time_share']:.2%}和"
        f"{point['positive_inventory_time_share']:.2%}。"
        "最小与最大实际库存、正负库存时间分别进入验证，且所有路径都保持在-10至+10手并在日终归零。"
        "这说明负库存由卖出成交真实形成，而不是报告把绝对库存误标成有符号库存。"
        "图展示全部21档gamma的负库存时间占比；该占比描述方向暴露，不替代库存RMS。"))
    visual("chart", "short_share")
    prose("policy_region", "理论市价区域与实际市价次数要分开看", (
        "图中的理论市价区域为剩余300秒、spread×库存状态网格上需要市价调整的比例，"
        "对19个日期和6个时段等权汇总。这直接检查论文提到的政策区域机制，不代表这些状态在回放中被访问的频率。"
        f"gamma=5与0.1的区域占比分别为{base['mean_full_horizon_impulse_state_share']:.1%}和"
        f"{point['mean_full_horizon_impulse_state_share']:.1%}；实际盘中市价次数分别为"
        f"{base['policy_market_orders']}和{point['policy_market_orders']}。"
        "大gamma可能扩大理论市价区域，同时因少挂单、少产生库存而减少实际次数。"))
    visual("chart", "impulse")
    prose("risk_detail", "状态区域与实际库存明细", (
        "下表并列理论市价区域、实际市价调整次数、库存RMS和持仓时间占比；"
        "各列衡量不同机制，不能把实际市价次数当作理论区域大小的替代指标。"
        "每个日期、时段、spread和库存的具体报价动作与市价目标另存政策状态表。"))
    visual("table", "inventory")
    prose("empirical_frontier", "历史数据给出经验风险收益点，不等同于论文模拟前沿", (
        "每个点来自同一组19个交易日：横轴为每日净损益样本标准差，纵轴为平均每日净损益，单位均为港元。"
        "散点用于观察gamma导致的风险与收益变化；重合点可来自多个gamma产生相同的回放路径。"
        "这些日期不是独立Monte Carlo路径，也未完成原论文基准收益平移，故不把它称为表6复现，"
        "不计算论文NIR或将19天结果年化。完整数值可由汇总CSV核对。"))
    visual("chart", "frontier")
    prose("execution_sensitivity", "Touch 检查执行模型对gamma结论的影响", (
        f"gamma=0.1在touch下净收益为HKD {variants['gamma_0.1']['aggregate']['touch']['net_pnl_hkd']:+,.2f}。"
        "下表保留所有gamma的touch结果。touch与through各自用历史成交证据重新校准强度并求解策略，"
        "所以两者属于执行模型敏感性，不能把差值解释成仅改变价格相等条件的净影响。"))
    visual("table", "touch")
    prose("next", "下一步围绕论文机制验证gamma，而非换策略", (
        "1. 将空头时间占比、最小库存和借券可用性加入上线前约束，再预先规定gamma比较范围。\n"
        "2. 沿当前原GP-QVI方法检查不同gamma下的市价控制阈值，以及限价成交后的价格变动，核查逆向选择。\n"
        "3. 在未参与本次研究的数据重放，检查风险收益权衡能否保持；不把本月网格上的最大收益当作已验证最优gamma。"))
    prose("questions", "仍需回答的研究问题", (
        "在固定成交代理下，库存风险降低能否抵消市价费用？较小gamma的交易现金流亏损是否集中在限价成交后的不利价格变动？"
        "负库存暴露在加入真实借券费、可借数量和强平规则后是否仍可接受？"
        "若按论文原始库存与费用单位重建模拟，当前的gamma映射能否重现其政策区域和风险收益趋势？"), None)
    prose("caveats", "工程复现与原论文的边界", (
        "- 本轮只改变gamma，所有档位使用原0906求解器；每侧固定100股，市价仅减库存。"
        "原论文还优化限价数量与一般市价冲击控制，本实现是其有限状态工程近似。\n"
        "- 3秒离散步中的运行成本按步首库存计算，区间内新成交和先买后卖的暂时库存未连续积分进Bellman；"
        "回放市价决策仅在报价节点评估。新增暴露计时揭示实际持仓风险，但没有改写原求解目标，"
        "高gamma结果饱和须结合这些离散近似解释。\n"
        "- 真实数据回放和论文的300秒Monte Carlo实验不同，不能据此宣布复现原表数值或原最优gamma。\n"
        "- 保留日级合法tick识别、原成交代理及费用；负库存按卖出成交现金流处理，但不包含借券可用性、借券费、"
        "强平规则、虚拟订单排队、隐藏流动性、最低佣金和市场冲击。因此结果不能直接视为可执行做空收益。\n"
        "- gamma=5与原0906逐日一致；全部21档与0913的成交、库存峰值和现金损益均逐日一致。"
        "新增正负暴露计时不改变现金流，全部gamma通过有符号库存边界、积分、生命周期、费用与归零检查。\n"
        "- 每档校准仅用过去日期，但比较同月结果属于敏感性研究；本轮没有输出按本月收益挑选的最优参数。"))
    artifact = {"surface": "report", "manifest": {
        "version": 1, "surface": "report", "title": TITLE,
        "description": "论文均值—二次库存惩罚GP-QVI方法：gamma、库存暴露与风险收益",
        "generatedAt": result["as_of"], "cards": [], "charts": charts, "tables": tables,
        "sources": sources, "blocks": blocks},
        "snapshot": {"version": 1, "generatedAt": result["as_of"], "status": "ready", "datasets": datasets},
        "sources": sources, "package_info": {}}
    (OUT / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
    notes = {"audience": "technical", "delivery": "portable html",
        "required_structure_map": {
            "Title": "artifact title and first markdown block",
            "Technical summary": "Technical Summary",
            "Key findings with visual evidence": "activity, economics, inventory, signed inventory, policy region, frontier",
            "Scope, data, and metric definitions": "样本、库存域与指标定义",
            "Methodology": "论文中的gamma控制的是收益与持仓风险的权衡; units",
            "Limitations, uncertainty, and robustness checks": "touch sensitivity; 工程复现与原论文的边界",
            "Recommended next steps": "下一步围绕论文机制验证gamma，而非换策略",
            "Further questions": "仍需回答的研究问题"
        },
        "chart_map": {
            "fills": {"question": "gamma如何影响maker成交", "family": "comparison", "type": "bar", "fields": ["gamma_label", "fills"], "takeaway": "低gamma显著增加成交", "palette": "single-root", "dataset": "anchors"},
            "inventory": {"question": "gamma如何影响时间加权库存", "family": "ordered relationship", "type": "line", "fields": ["gamma_label", "inventory_rms"], "takeaway": "低gamma提高库存RMS", "palette": "single-root", "dataset": "anchors"},
            "short_share": {"question": "各gamma实际有多少时间处于负库存", "family": "comparison", "type": "bar", "fields": ["gamma_label", "negative_inventory_time_pct"], "takeaway": "所有gamma均访问负库存", "palette": "single-root", "dataset": "through"},
            "impulse": {"question": "gamma如何影响理论市价减仓区域", "family": "comparison", "type": "bar", "fields": ["gamma_label", "impulse_state_pct"], "takeaway": "高gamma扩大理论减仓区域", "palette": "single-root", "dataset": "anchors"},
            "frontier": {"question": "经验收益与日收益波动如何共同变化", "family": "relationship", "type": "scatter", "fields": ["daily_net_std", "mean_daily_net"], "takeaway": "低gamma未改善本样本风险收益", "palette": "single-root", "dataset": "through"}
        },
        "analysis_validation": result["validation"], "gamma_selected_using_monthly_pnl": False,
        "only_variable": "gamma", "paper_sections": ["3.1–3.3", "4 / Table 4 / Table 6"],
        "units": "lot coefficient is 10000 times share coefficient for the same cash currency",
        "original_model_unchanged": True,
        "signed_inventory_condition": "explicit symmetric domain [-10, 10] lots",
        "omitted_visuals": []}
    (OUT / "report_source_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2))
    config = variants["gamma_5"]["config"].copy()
    config.pop("inventory_penalty_gamma")
    (OUT / "study_config.json").write_text(json.dumps({
        "method": "original gp_qvi", "market_orders_enabled": True, "fixed_config": config,
        "inventory_domain_lots": [-config["max_inventory_lots"], config["max_inventory_lots"]],
        "negative_inventory_allowed": True,
        "gamma_grid": [v["config"]["inventory_penalty_gamma"] for v in variants.values()],
        "gamma_selection": "none; sensitivity only", "variable": "inventory_penalty_gamma",
    }, ensure_ascii=False, indent=2))
    matches = sorted(Path.home().glob(".codex/plugins/cache/openai-curated-remote/data-analytics/*/package.json"), reverse=True)
    plugin_root = Path(os.environ["DATA_ANALYTICS_PLUGIN_ROOT"]).expanduser() if os.environ.get("DATA_ANALYTICS_PLUGIN_ROOT") else matches[0].parent
    proc = subprocess.run(["node", str(HERE / "deliver_report.mjs"), str(plugin_root),
                           str(OUT / "artifact.json"), str(OUT / "report.html")],
                          text=True, capture_output=True, cwd=plugin_root)
    (OUT / "report_delivery.log").write_text(proc.stdout + proc.stderr)
    print(proc.stdout, proc.stderr)
    proc.check_returncode()


if __name__ == "__main__":
    main()
