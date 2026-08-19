#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "../..");
const artifactPath = path.join(root, "output/report/artifact_reproduction_summary_zh.json");
const resultPath = path.join(root, "output/results/7709_state_example.json");
const imagePath = path.join(root, "output/report/7709_two_states_depth.png");

const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
const example = JSON.parse(fs.readFileSync(resultPath, "utf8"));
const imageDataUri = `data:image/png;base64,${fs.readFileSync(imagePath).toString("base64")}`;
const generatedAt = new Date().toISOString();

const newBlockIds = new Set([
  "state_example_heading",
  "state_example_figure",
  "state_example_table_block",
  "state_example_interpretation",
]);
artifact.manifest.blocks = artifact.manifest.blocks.filter((block) => !newBlockIds.has(block.id));

const architectureTableIndex = artifact.manifest.blocks.findIndex(
  (block) => block.id === "deeplob_architecture_table_block",
);
if (architectureTableIndex < 0) {
  throw new Error("Could not find the architecture table insertion point");
}

const stateBlocks = [
  {
    id: "state_example_heading",
    type: "markdown",
    sourceId: "state_example_source",
    body: "## 一个 state 就是一张完整十档盘口快照\n\n下面用 7709 的两张真实采样快照说明模型到底看到了什么。每张 state 包含 **10 档 × 4 个字段 = 40 个值**，排列顺序固定为：\n\n```text\n[ask_price_1, ask_size_1, bid_price_1, bid_size_1,\n ask_price_2, ask_size_2, bid_price_2, bid_size_2,\n ...\n ask_price_10, ask_size_10, bid_price_10, bid_size_10]\n```\n\n图中的 `t` 为 **2026-07-09 10:10:08.341**，`t+1` 为下一张采样快照 **10:10:08.372**，两者相隔 **31 ms**。这里的 `t+1` 不是固定 1 ms 后，也不是仅发生了一条原始委托事件；7709 处理程序每累计 **10 个有效盘口更新组**保存一张快照，因此两张采样快照之间可以有多处档位同时变化。",
  },
  {
    id: "state_example_figure",
    type: "html",
    sourceId: "state_example_source",
    body: `<figure aria-labelledby="state-example-caption" style="margin:0"><img src="${imageDataUri}" alt="7709 在 t 和 t+1 的真实十档买卖盘口深度图" style="display:block;width:100%;height:auto;border-radius:12px;border:1px solid #e4e7ec;background:#fff"><figcaption id="state-example-caption" style="margin-top:10px;color:#475467;font-size:0.92rem;line-height:1.55">蓝色柱为各买价上的委托数量，橙色柱为各卖价上的委托数量，虚线为中间价。上下两图使用相同的纵轴尺度；价格越靠近中间价，档位越接近 L1。</figcaption></figure>`,
  },
  {
    id: "state_example_interpretation",
    type: "markdown",
    sourceId: "state_example_source",
    body: "这两张 state 中，最优买卖价从 `0.8976 / 0.8978` 变为 `0.8984 / 0.8988` 港元，中间价从 **0.8977** 变为 **0.8986** 港元。以 state `t` 的 L1 为例，实际盘口是 Ask `0.8978 × 3,400`、Bid `0.8976 × 700`；送入当前模型文件的四值切片则是 `[0.8978, 0.0340, 0.8976, 0.0070]`，因为数量按 **100,000** 缩放。完整输入不是单独这一行，而是把表中 L1 至 L10 的十个四值切片依次拼成 40 维 state，再把最近 100 个 state 堆成 `100 × 40` 的输入窗口。",
  },
];
artifact.manifest.blocks.splice(architectureTableIndex + 1, 0, ...stateBlocks);

artifact.manifest.tables = artifact.manifest.tables.filter(
  (table) => table.id !== "state_example_table",
);

const source = {
  id: "state_example_source",
  label: "7709 两张相邻采样快照",
  path: "output/results/7709_state_example.json",
};
artifact.manifest.sources = artifact.manifest.sources.filter((item) => item.id !== source.id);
artifact.manifest.sources.push(source);
artifact.sources = artifact.sources.filter((item) => item.id !== source.id);
artifact.sources.push({
  id: source.id,
  path: source.path,
  query: {
    description: "从 7709 重建盘口中核对两张相邻采样快照、31 ms 间隔、十档价量和数量缩放。",
    executed_at: generatedAt,
  },
});

delete artifact.snapshot.datasets.state_example_rows;
artifact.manifest.generatedAt = generatedAt;
artifact.snapshot.generatedAt = generatedAt;

fs.writeFileSync(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`);
console.log(`Updated ${artifactPath}`);
console.log(`Embedded state figure with ${imageDataUri.length} image URI characters`);
