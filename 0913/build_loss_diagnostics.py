#!/usr/bin/env python3
"""Build the scientific loss readout from reconciled replay diagnostics."""
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "output/loss_diagnostics"
sys.path.insert(0, str(ROOT / "0906"))
sys.path.append(str(ROOT / "0824"))
import gp_qvi_model as gp


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    validation = json.loads((OUT / "validation.json").read_text())
    assert validation["all_passed"] and validation["runs"] == 190
    rows = read_csv(OUT / "summary.csv")
    lookup = {(float(r["gamma"]), r["fill_mode"]): r for r in rows}
    slices, context, manifests = defaultdict(list), [], []
    for path in sorted((OUT / "events").glob("*.csv.gz")):
        date = path.name[:10]
        book_path = ROOT / f"0824/data/july_2026/hk07709_{date}_mbo_tolerant_5m_s10.npz"
        with np.load(book_path) as data:
            book, segments = data["features"], data["segments"]
            times = gp.compact_time_to_day_ms(data["send_times"])
        mids = (book[:, 0].astype(float) + book[:, 2].astype(float)) / 2
        spreads = book[:, 0] - book[:, 2]
        context.append({"date": date, "mid_first": float(mids[0]), "mid_last": float(mids[-1]),
                        "mid_change_pct": float(100 * (mids[-1] / mids[0] - 1)),
                        "locked_crossed_share": float(np.mean(spreads <= 0)),
                        "median_spread_hkd": float(np.median(spreads)), "median_mid_hkd": float(np.median(mids))})
        with gzip.open(path, "rt") as stream:
            for event in csv.DictReader(stream):
                if float(event["gamma"]) != .01 or event["kind"] != "maker":
                    continue
                target_time = int(event["time_ms"]) + 3000
                index = int(np.searchsorted(times, target_time))
                if (index >= len(times) or int(segments[index]) != int(event["segment"])
                        or times[index] - target_time > 1000 or int(event["snapshot_age_ms"]) > 1000):
                    continue
                signed = int(event["signed_lots"])
                price = float(event["price_hkd"])
                quote_index = int(event["quote_index"])
                action = "best" if abs(float(book[quote_index, 2 if signed > 0 else 0]) - price) < 1e-5 else "improve"
                value = signed * gp.LOT_SIZE * (mids[index] - price)
                for dimension, label in (("side", "buy" if signed > 0 else "sell"), ("source", event["source"]),
                                         ("action", action), ("date", date)):
                    slices[(event["fill_mode"], dimension, label)].append(value)
    slice_rows = [{"fill_mode": mode, "dimension": dimension, "label": label, "count": len(values),
                   "mean_3s_gross_markout_hkd": float(np.mean(values))}
                  for (mode, dimension, label), values in slices.items()]
    write_csv(OUT / "markout_slices.csv", slice_rows)
    write_csv(OUT / "market_context.csv", context)
    for date in ["2026-07-02"] + [r["date"] for r in context]:
        for suffix in ("mbo_tolerant_5m_s10", "trades"):
            path = ROOT / f"0824/data/july_2026/hk07709_{date}_{suffix}.npz"
            manifests.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT / "input_manifest.json").write_text(json.dumps(manifests, indent=2))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.6), sharey=True)
    labels = ["Quote-time\nedge", "Quote-to-fill\nrepricing", "Inventory\ncarry", "Market / terminal\nexecution", "Trading\nfees", "Net\nP&L"]
    plot_data = []
    for axis, mode in zip(axes, gp.FILL_MODES):
        row = lookup[(.01, mode)]
        deltas = [float(row["maker_quote_edge_hkd"]), float(row["maker_quote_to_fill_hkd"]),
                  float(row["inventory_carry_hkd"]), float(row["market_execution_edge_hkd"]) + float(row["terminal_execution_edge_hkd"]),
                  -float(row["all_in_fees_hkd"])]
        net = float(row["net_pnl_hkd"])
        assert math.isclose(sum(deltas), net, abs_tol=1e-5)
        running = 0.0
        for i, value in enumerate(deltas):
            start, end = running / 1000, (running + value) / 1000
            axis.bar(i, abs(value) / 1000, bottom=min(start, end), width=.65,
                     color="#4C78A8" if value >= 0 else "#DD8452", edgecolor="#42464D", linewidth=.5)
            axis.text(i, max(start, end) + 4, f"{value / 1000:+,.2f}", ha="center", va="bottom", fontsize=10)
            if i < 4:
                axis.plot([i + .325, i + 1 - .325], [end, end], color="#AAAAAA", linewidth=.8)
            running += value
            plot_data.append({"mode": mode, "component": labels[i].replace("\n", " "), "value_hkd": value})
        axis.bar(5, abs(net) / 1000, bottom=min(0, net / 1000), width=.65, color="#42464D")
        axis.text(5, min(0, net / 1000) - 4, f"{net / 1000:+,.2f}", ha="center", va="top", fontweight="bold")
        axis.axhline(0, color="#42464D", linewidth=.8)
        axis.set_ylim(-110, 115)
        axis.set_xticks(range(6), labels)
        axis.set_ylabel("HKD thousands")
        axis.set_title(f"{mode.upper()}  |  {int(row['maker_fills']):,} maker fills", loc="left", pad=12, fontweight="bold")
        axis.yaxis.grid(True, color="#E8E8E8", linewidth=.6)
        axis.set_axisbelow(True)
    fig.suptitle("GP replay P&L decomposition: gamma = 0.01", x=.075, ha="left", fontsize=17, fontweight="bold", y=.99)
    fig.text(.075, .948, "19 test days, July 2026 | 100 shares per lot | 2.77 bps fee per execution side", fontsize=11)
    fig.text(.075, .017, "Source: reconciled event logs and summary.csv. Sampled L1 midpoints; touch/through are proxy fills.\nComponents add to net P&L. Inventory penalty is not deducted from cash P&L.", fontsize=9, color="#555555")
    fig.subplots_adjust(top=.885, bottom=.13, left=.075, right=.985, hspace=.42)
    fig.savefig(OUT / "loss_decomposition.png", dpi=180)
    plt.close(fig)
    write_csv(OUT / "plot_data.csv", plot_data)

    lines = ["# GP 回测亏损诊断", "", "_2026-09-17 · 19 个测试交易日 · 历史盘口代理回放_", "", "---", "",
        "本次重新回放 5 个 gamma × 2 个成交口径 × 19 天，共 190 组；新增日志不改变报价、成交或现金流。",
        "主要发现：默认 gamma=5 的毛收益不足以覆盖费用；低 gamma 在 through 下还存在显著的报价到成交的不利移动和库存持有亏损。",
        "这些结果支持当前策略与成交环境不匹配，不能推断所有做市策略普遍亏损，更不能从单月结果估计行业亏损概率。", "",
        "## 📊 毛收益与费用", "", "| gamma | 成交口径 | 毛收益 HKD | 费用 HKD | 净收益 HKD | 盈利日/19 |",
        "|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {float(row['gamma']):g} | {row['fill_mode']} | {float(row['gross_pnl_hkd']):,.2f} | {float(row['all_in_fees_hkd']):,.2f} | {float(row['net_pnl_hkd']):,.2f} | {int(row['profitable_days'])} |")
    lines += ["", "来源：[summary.csv](summary.csv)。净收益为现金毛收益减费用；gamma 的库存惩罚没有从净收益中扣除。",
        "费用固定为代码假设的单边 2.77bp，包括 1.27bp 固定费用与 1.50bp 经纪费用；不是本次独立核实的实际账户费率。",
        "固定交易路径只去掉全部费用时，gamma=0.01/through 仍亏 49,524.42 HKD；touch 毛收益为 +1,369.46 HKD。",
        "固定路径只保留代码的 1.27bp 费用时，两者净收益分别为 -67,518.50、-27,069.06 HKD。这里没有按新费率重新优化策略。", "",
        "## 🔬 逐笔损益拆解", "", "![两种成交口径的损益分解](loss_decomposition.png)", "",
        "取 gamma=0.01/through，拆解如下：", "", "| 分项 | HKD |", "|---|---:|"]
    row = lookup[(.01, "through")]
    for label, field in (("成交挂单相对报价时中间价的优势", "maker_quote_edge_hkd"),
                         ("报价到成交期间的中间价变化", "maker_quote_to_fill_hkd"),
                         ("已有库存持有期间的中间价变化", "inventory_carry_hkd"),
                         ("盘中市价单相对中间价的成交损益", "market_execution_edge_hkd"),
                         ("日终平仓相对中间价的成交损益", "terminal_execution_edge_hkd")):
        lines.append(f"| {label} | {float(row[field]):+,.2f} |")
    lines += [f"| 交易费用 | {-float(row['all_in_fees_hkd']):+,.2f} |", f"| 净收益 | {float(row['net_pnl_hkd']):+,.2f} |", "",
        "上述各项严格相加等于现金净收益。报价时优势是相对当时中间价的账面距离，不是实际已实现收益，也不是已验证的条件期望。",
        f"报价到成交的不利移动抵消了报价时优势的 {abs(float(row['maker_quote_to_fill_hkd'])) / float(row['maker_quote_edge_hkd']):.1%}。",
        "逐笔中间价使用成交时间之前最近一条快照。分解会随选用的标记价格改变，但总现金损益不变。",
        "成交事件 CSV.gz 见 [events](events/)，每条含报价中间价、成交价、成交时可见中间价、库存和费用。", "",
        "以买入为正，事件成交数量为 Δq，成交前库存为 q，中间价为 m、成交价为 p，单笔数量单位为 100 股：", "",
        "```text", "毛损益 = Σ 100·Δq·(m_event − p) + Σ 100·q_before·(m_event − m_previous_event)",
        "限价成交项 = Σ 100·Δq·(m_quote − p) + Σ 100·Δq·(m_event − m_quote)",
        "净损益 = 毛损益 − 费用（每日初始及终止库存均为零）", "```", "",
        "## 📉 成交后的价格证据", "",
        "成交后 markout 定义为：买单 (未来中间价−成交价)×100；卖单 (成交价−未来中间价)×100。",
        "它用于观察成交质量，不能再加到上面的损益拆解中，否则会重复计量。", "",
        "| 口径 | 目标间隔 秒 | 样本成交数 | 平均毛 markout/笔 HKD | 扣单次成交费后 HKD |",
        "|---|---:|---:|---:|---:|"]
    for mark in read_csv(OUT / "markouts.csv"):
        if float(mark["gamma"]) == .01:
            lines.append(f"| {mark['fill_mode']} | {mark['horizon_seconds']} | {int(mark['count']):,} | {float(mark['gross_markout_per_fill_hkd']):+.3f} | {float(mark['net_markout_per_fill_hkd']):+.3f} |")
    lines += ["", "取目标时间之后第一条、同交易段且晚到不超过 1 秒的快照；剔除成交时可见中间价已陈旧超过 1 秒的样本。",
        "上表没有计未来平仓的手续费或跨价差成本。来源：[markouts.csv](markouts.csv)。", "",
        "gamma=0.01/through 的 3 秒毛 markout 在 19 个测试日全部为负；买单平均 -2.335 HKD，卖单 -1.796 HKD。",
        "这说明不利成交并非仅表现为单一方向持仓亏损。它与逆向选择相符，但成交代理本身也会筛选不利路径，不能当作真实账户成交质量的验证。",
        "来源：[markout_slices.csv](markout_slices.csv)。", "",
        "## ⚙️ 模型与回放的差异", "",
        "```mermaid", "flowchart LR", "    accTitle: GP 报价与回放成交差异",
        "    accDescr: 动态规划按当前价差估计成交收益，而历史回放保持绝对报价直到成交或到期，期间价格变化影响成交质量。",
        '    quote["观察当前价差"] --> model["DP 计算价差收益与成交概率"]',
        '    model --> order["按策略挂单"]', '    order --> wait["保持价格至成交或 20 快照到期"]',
        '    wait --> selection["后续价格触碰或穿透时判成交"]',
        '    selection --> pnl["成交质量、库存损益及费用决定净收益"]', "```", "",
        "1. **报价更新不一致。** 求解器的即时收益按当前半价差减改善一档与费用计算；回放的绝对报价保持到 20 快照到期，首次单侧成交后也不立即重新优化剩余订单。旧报价随市场变化可能变得不利。",
        "2. **成交强度没有描述成交后的价格。** 用次数/暴露时间估计 lambda 只回答成交速度；它不估计成交后条件中间价变动。仅重新调 gamma 或加入独立 sigma 参数不会自动修复这个差异。",
        "3. **through 是筛选条件，不是真实成交记录。** 买单要求成交价或对手报价至少低一 tick，卖单反之；它倾向于只保留价格已经穿过挂单的不利路径。touch 也忽略真实队列位置，两者不是严格的利润上下界。",
        "4. **时间与状态近似。** DP 固定 3 秒一步，回放订单寿命取 20 个抽样快照，日内实际长度变化；成交校准把整段暴露归入初始 spread，而 GP 原始连续状态计时按状态占用时间累计。原代码还把 spread 转移和成交结果分开计算。",
        "5. **费用模型有业务假设。** 当前没有做市返佣或专属费率，也没有最低佣金、借券、排队和市场冲击。需确认实际交易资格和费用后再比较可交易性。",
        "6. **高 gamma 的离散近似。** 3 秒内运行惩罚按步首库存计；同一步先买后卖的短暂库存没有连续积分到 Bellman 风险项中。这有助于解释高 gamma 饱和，不应被当作精确连续时间 GP 复现。", "",
        "代码定位：[求解器](../../../0906/gp_qvi_model.py)、[回放](../../paper_gp_replay.py)。论文的均值惩罚化简见 §3.3；控制的报价随当时市场价格定义，见 §2.2。[GP 原文](https://arxiv.org/pdf/1106.5040)。", "",
        "## 🔍 数据与核验边界", "",
        "- 190 组回放的成交数量、报价数量、现金毛净收益、费用、库存峰值及库存积分均与原结果一致。每个测试日额外比较 gamma=5/through 开启和关闭日志的完整返回对象，完全一致。",
        "- 所有逐笔现金流、费用及上述毛损益恒等式对账通过；已有 16 项库存、成交和 GP 求解器测试通过。",
        "- 19 个测试日中 9 日在已有源审计中被标记过重建异常或间隙。只保留其余 10 个测试日，gamma=0.01/through 仍有毛亏 -23,196.17 HKD、净亏 -41,722.55 HKD。该子集的训练窗口仍可能包含被标记日期；这不是完整清洁数据重训。",
        "- 缓存 L1 没有发现锁盘/交叉报价，但这不证明订单簿重建完全正确；未独立核验交易所原始事件或实际账户成交。",
        "- 原回测用测试日全日价格网格识别 tick，严格实盘应换成事先已知的合约规则。本次保持原行为用于诊断，并未宣称整个回测已经完全消除前视问题。",
        "- 只有一个品种和一个月份。Gamma 参数在同月比较，不能据此声称有样本外最优参数或估计做市行业亏损概率。", "",
        "## 🎯 判断与下一步", "",
        "做市的收益要覆盖交易成本、库存价值变化等风险，价差不是无条件利润；BIS 对做市损益的解释也区分交易便利收入与库存收入。[BIS](https://www.bis.org/publications/economics-market-making)",
        "当前证据足以判定：高 gamma 的小额毛利润被费用吃掉；低 gamma 增加了本来就缺乏足够净优势的成交和库存风险。不能据此断言所有做市天然亏损。",
        "原 WoMO 基准在 through 下净赚约 58.48 HKD，但全月仅 10 笔限价成交；这是对“所有变体一直亏”的例外，不足以证明可持续盈利。", "",
        "建议先核验原始事件与成交代理，然后统一 DP 和回放的报价有效期、重报价及单侧成交后的处理；以历史训练窗估计成交后的条件价格变化和净优势。",
        "在未参与本轮参数比较的时间段测试这些调整。不要为获得正收益而直接放宽成交条件，也不要把 sigma 加回去当作现成解决办法。", "",
        "## 📁 复现与文件", "", "```bash", ".venv/bin/python 0913/diagnose_gp_losses.py",
        ".venv/bin/python 0913/build_loss_diagnostics.py", "```", "",
        "- [汇总损益](summary.csv)、[逐日损益](daily_decomposition.csv)、[逐笔日志](events/)",
        "- [成交后价格](markouts.csv)、[分组价格变化](markout_slices.csv)、[市场背景](market_context.csv)",
        "- [输入文件哈希](input_manifest.json)、[核验结果及代码哈希](validation.json)",
        "- 本轮仅增加可选观察日志与诊断脚本；原策略及成交规则保持一致。", ""]
    (OUT / "README.md").write_text("\n".join(lines))
    print("WROTE", OUT / "README.md")
    print("WROTE", OUT / "loss_decomposition.png")


if __name__ == "__main__":
    main()
