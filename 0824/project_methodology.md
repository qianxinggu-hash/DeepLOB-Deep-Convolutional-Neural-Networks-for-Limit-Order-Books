# 从 DeepLOB 到方向增强做市：HK 07709 项目思路、公式与数学原理

## 技术摘要

本项目研究的问题不是“能否用一个分类器直接预测涨跌赚钱”，而是：**限价订单簿中是否存在
可跨时段、跨交易日复现的短期方向信息，以及这些信息能否作为传统做市模型之外的附加状态，
改善报价、库存管理和逆向选择。**

研究路线经历了三个阶段：

1. **DeepLOB 阶段**：从十档限价订单簿的100步序列学习上涨、平稳、下跌三分类。修复盘口
   重建后，初始 DeepLOB 仍出现训练 Macro-F1 94.01%、测试 Macro-F1 32.37%的严重过拟合；
   加入严格时间验证、正则化、早停和因果辅助特征后，测试 Macro-F1 提升到48.26%。
2. **逻辑回归阶段**：用25个只依赖当前及过去盘口的显式特征训练多项逻辑回归。单日测试
   Accuracy/Macro-F1 达到60.63%/60.94%，完整交易日 walk-forward 仍约58%–63%。这说明
   盘口中存在可预测方向信息，但不代表信息幅度足以覆盖点差和费用。严格来说，**“单独用
   逻辑回归预测涨跌即可赚钱”的逻辑目前不成立**：模型优化的是分类损失，不是可执行净收益，
   而严格跨日主动交易回测即使在未计额外费用时仍为负。
3. **方向增强做市阶段**：不再把分类结果当作立即吃单指令，而是把
   \(p(\mathrm{up})-p(\mathrm{down})\) 校准为预期中价移动，加入 Avellaneda–Stoikov（AS）
   /Guilbaud–Pham（GP）式报价中心，同时保留库存惩罚、离散 tick、成交概率和日末平仓。
   公平对比将方向+库存、验证期独立调参的经典 AS 和 GP-style 离散库存基准放在同一数据、
   费用与成交假设下。方向+库存在 MBO 两日合计优于两个基准，但在 MBP 两日合计反而最差；
   5/10/20秒 markout 也没有在所有日期和买卖侧同时改善。

因此当前结论是：**方向预测已经表现出“做市成交过滤器”的增量价值，但尚未证明能够形成
跨口径、跨市场状态稳定盈利的策略。** 下一阶段应把研究目标从分类 Accuracy 进一步改为
买卖两侧条件成交概率、条件 markout 和净 maker edge 的联合估计。

---

## 1. 研究问题与总体逻辑

传统 AS 类做市模型通常把参考价格写成零漂移扩散：

$$
dS_t = \sigma\,dW_t,
$$

其中 \(W_t\) 是标准布朗运动，意味着在当前信息集下：

$$
\mathbb{E}[S_{t+h}-S_t\mid\mathcal{F}_t]=0.
$$

这个假设方便求解库存风险和最优报价，但忽略了订单簿可能包含的短期方向信息。如果十档深度、
价差、排队不平衡和最近价格路径使得

$$
\mu_t(h)=\mathbb{E}[S_{t+h}-S_t\mid X_t]\neq 0,
$$

那么做市商不应在中价两侧完全对称地报价。上涨概率较高时，卖单更容易遭遇逆向选择，买单
则可能获得更好的后续 markout；下跌时反向。

项目的核心链路因此是：

```text
逐事件行情
  → 严格恢复十档 LOB
  → 只用过去信息构造样本和标签
  → DeepLOB / 逻辑回归估计方向概率
  → 概率差校准为预期 tick 移动
  → 与库存惩罚共同移动做市报价中心
  → 用真实成交打印、费用和秒级 markout 评价
```

这里所说的“因果特征”是 **causal-in-time**：特征在时刻 \(t\) 已经可见，不使用 \(t\) 之后
的信息。它保证预测时序上的因果方向，但**不等于**已经识别经济学或统计学意义上的因果效应。

---

## 2. 从逐笔事件恢复限价订单簿

### 2.1 两种数据口径

- **MBO（Market-by-Order）**：逐订单事件。需要维护
  `OrderID → (side, price, remaining quantity)`，再按价格聚合。
- **MBP（Market-by-Price）**：逐价位绝对更新。每条消息直接给出该方向、该价格的最新聚合
  数量，更新时覆盖旧值。

设买卖方向 \(s\in\{b,a\}\)，订单 \(i\) 的价格和剩余数量为 \(p_i,q_i\)。MBO 的聚合深度为：

$$
Q_t^s(p)=\sum_{i\in\mathcal O_t}q_{i,t}
\mathbf 1\{s_i=s,\ p_i=p\}.
$$

MBO 三类状态转移为：

$$
\begin{aligned}
\text{Add:}\quad &Q_t^s(p)=Q_{t^-}^s(p)+q_i,\\
\text{Modify:}\quad &Q_t^s(p)=Q_{t^-}^s(p)+(q_i^{new}-q_i^{old}),\\
\text{Delete:}\quad &Q_t^s(p)=Q_{t^-}^s(p)-q_i^{remaining}.
\end{aligned}
$$

MBP 更新则是：

$$
Q_t^s(p)\leftarrow q_t^{reported}(s,p),
$$

其中 \(q=0\) 表示删除价位，而不是在旧数量上加零。

### 2.2 严格重放原则

1. 按交易所/发送时间的原始顺序重放；时间倒序时停止或污染状态。
2. 同一毫秒或同一 `SendTime` 的消息原子应用完毕后才采样。
3. MBO 中重复新增、修改未知订单、删除未知订单、方向不一致或负档位数量都会污染状态；
   污染后不再猜测性修复。
4. MBP 的成交事件不重复扣减深度，因为绝对档位更新已经表示成交后的数量。
5. `CLEAR`、午休和重同步切断序列片段；输入窗口和未来标签不能跨片段。
6. 不足十档、交叉或锁定盘口只跳过观察，不通过删除“疑似过期订单”把盘口强行修好。

输出十档状态向量：

$$
x_t=(a_{1,t},v^a_{1,t},b_{1,t},v^b_{1,t},\ldots,
a_{10,t},v^a_{10,t},b_{10,t},v^b_{10,t})\in\mathbb R^{40}.
$$

其中中价和价差为：

$$
m_t=\frac{a_{1,t}+b_{1,t}}{2},\qquad
s_t=a_{1,t}-b_{1,t}.
$$

实现位于 [`lob_reconstruction.py`](lob_reconstruction.py)。详细异常审计见
[`reconstruction_and_results.md`](reconstruction_and_results.md) 第1–4节和第11.1节。

---

## 3. 标签：我们究竟预测什么

### 3.1 DeepLOB 风格三分类标签

固定 horizon \(k=20\)。定义过去和未来中价均值：

$$
\bar m_t^{-}=\frac{1}{k}\sum_{i=0}^{k-1}m_{t-i},\qquad
\bar m_t^{+}=\frac{1}{k}\sum_{i=1}^{k}m_{t+i}.
$$

对称收益标签变量为：

$$
r_t=\frac{\bar m_t^{+}}{\bar m_t^{-}}-1.
$$

阈值 \(\alpha\) 只用训练区间的 \(|r_t|\) 三分之一分位数拟合：

$$
\alpha=Q_{1/3}\left(\{|r_t|:t\in\mathcal T_{train}\}\right).
$$

三分类标签为：

$$
y_t=
\begin{cases}
0,&r_t<-\alpha \quad(\text{down}),\\
1,&|r_t|\le\alpha \quad(\text{stationary}),\\
2,&r_t>\alpha \quad(\text{up}).
\end{cases}
$$

这个标签平滑了逐 tick 噪声，但存在一个重要边界：它比较“未来均价”和“过去均价”，并不
直接等于从可执行入场价到未来退出价的净收益。因此分类正确可能仍无法覆盖点差和费用。

### 3.2 严格时间切分

若切点为 \(T_{split}\)，训练目标必须满足未来标签仍在切点前；测试目标的完整100步输入
必须从切点后开始：

$$
t+k<T_{split}\quad(\text{train}),
$$

$$
t-L+1\ge T_{split}\quad(\text{test}),\qquad L=100.
$$

这避免训练标签偷看测试期，也避免测试输入与训练期共享盘口状态。完整交易日 walk-forward
进一步要求模型训练日期严格早于测试日期。

---

## 4. 第一阶段：DeepLOB

### 4.1 输入变换

DeepLOB 每个样本使用最近 \(L=100\) 个十档状态：

$$
X_t=[x_{t-L+1},\ldots,x_t]\in\mathbb R^{100\times40}.
$$

为减少价格水平和数量长尾的影响，每个时刻的档位价格相对当前中价转换为基点：

$$
\tilde p_{\ell,t}=\left(\frac{p_{\ell,t}}{m_t}-1\right)10^4,
$$

数量使用：

$$
\tilde v_{\ell,t}=\log(1+v_{\ell,t}).
$$

随后按列 z-score：

$$
z_{j,t}=\frac{\tilde x_{j,t}-\mu_j^{train}}{\sigma_j^{train}},
$$

均值和标准差只在训练前缀拟合。

### 4.2 网络结构与数学含义

当前实现的 DeepLOB 包含：

1. **卷积模块**：在价格/数量和时间维度提取局部盘口形状与短期事件模式；
2. **Inception 模块**：并行使用时间核3、时间核5和池化分支，提取不同时间尺度的变化；
3. **LSTM**：把卷积后的多尺度特征作为序列输入，用隐藏状态总结较长依赖；
4. **线性分类器**：由最终64维 LSTM 隐状态输出三个 logits。

抽象表示为：

$$
h_t^{cnn}=f_{CNN}(X_t),
$$

$$
h_t^{inc}=\operatorname{concat}
\left(f_3(h_t^{cnn}),f_5(h_t^{cnn}),f_{pool}(h_t^{cnn})\right),
$$

$$
h_t^{lstm}=f_{LSTM}(h_{t-L+1:t}^{inc}),
$$

$$
o_t=Wh_t^{lstm}+b,\qquad
p_{t,k}=\frac{e^{o_{t,k}}}{\sum_{j=0}^{2}e^{o_{t,j}}}.
$$

使用交叉熵损失：

$$
\mathcal L_{CE}=-\frac1N\sum_{i=1}^{N}\sum_{k=0}^{2}
\mathbf 1(y_i=k)\log p_{i,k}.
$$

改进模型加入 label smoothing \(\varepsilon=0.03\)，将 one-hot 目标替换为：

$$
\tilde y_{i,k}=(1-\varepsilon)\mathbf 1(y_i=k)+\frac{\varepsilon}{3},
$$

并使用 dropout、AdamW、weight decay、梯度裁剪和内部时间验证早停。

### 4.3 为什么 DeepLOB 过拟合

初始实验结果：

| 模型 | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|
| 初始 DeepLOB，12轮 | 33.43% | 32.52% | 32.37% |
| 恒预测平稳 | 41.60% | 33.33% | 19.59% |
| 改进 DeepLOB，验证早停 | 48.01% | 50.18% | 48.26% |

初始 DeepLOB 的训练 Macro-F1 达94.01%，测试却只有32.37%。主要原因是：

- 相邻100步窗口高度重叠，名义样本数远大于有效独立样本数；
- 单日早段和晚段的价格、波动率、价差及标签比例发生漂移；
- 网络容量相对当前独立日数量过大；
- 逐时刻中价归一化保留盘口形状，却弱化了与标签有关的绝对短期价格路径；
- 初版没有在训练前缀内部做时间验证，12轮是在追逐训练拟合。

从偏差—方差角度看，DeepLOB 具有较低表示偏差但较高估计方差。在数据天数少、窗口高度
相关和分布漂移明显时，复杂网络容易学习单日时段特征而不是可跨日结构。

---

## 5. 第二阶段：25维因果逻辑回归

### 5.1 特征设计

逻辑回归使用时刻 \(t\) 已经可见的25维特征 \(z_t\)：

#### 价格动量

对 \(h\in\{1,2,5,10,20,50,100\}\)：

$$
R_t(h)=\left(\frac{m_t}{m_{t-h}}-1\right)10^4.
$$

#### 相对移动均价位置

对 \(h\in\{5,20,50\}\)：

$$
MA_t(h)=\frac1h\sum_{i=0}^{h-1}m_{t-i},\qquad
P_t(h)=\left(\frac{m_t}{MA_t(h)}-1\right)10^4.
$$

#### 相对价差

$$
Spread_t=\frac{a_{1,t}-b_{1,t}}{m_t}10^4.
$$

#### 累计深度不平衡

对 \(L\in\{1,2,3,5,10\}\)：

$$
I_t(L)=
\frac{\sum_{\ell=1}^{L}v^b_{\ell,t}-\sum_{\ell=1}^{L}v^a_{\ell,t}}
{\sum_{\ell=1}^{L}v^b_{\ell,t}+\sum_{\ell=1}^{L}v^a_{\ell,t}+\epsilon}.
$$

#### 逐档排队不平衡

对前五档：

$$
I_{\ell,t}=
\frac{v^b_{\ell,t}-v^a_{\ell,t}}
{v^b_{\ell,t}+v^a_{\ell,t}+\epsilon}.
$$

此外还包括买卖侧一、二档之间的相邻档距：

$$
G_t^a=\frac{a_{2,t}-a_{1,t}}{m_t}10^4,\qquad
G_t^b=\frac{b_{1,t}-b_{2,t}}{m_t}10^4,
$$

以及十档总深度：

$$
D_t^a=\log\left(1+\sum_{\ell=1}^{10}v^a_{\ell,t}\right),\qquad
D_t^b=\log\left(1+\sum_{\ell=1}^{10}v^b_{\ell,t}\right).
$$

### 5.2 多项逻辑回归

标准化后的特征为 \(\hat z_t\)。对类别 \(k\in\{0,1,2\}\)：

$$
p(y_t=k\mid z_t)=
\frac{\exp(w_k^\top\hat z_t+b_k)}
{\sum_{j=0}^{2}\exp(w_j^\top\hat z_t+b_j)}.
$$

带类别权重和 L2 正则的目标可写为：

$$
\min_{W,b}
-\sum_{i=1}^{N}\omega_{y_i}\log p(y_i\mid z_i)
+\frac{1}{2C}\|W\|_F^2,
$$

其中 \(\omega_k\) 与训练类别频数近似成反比，\(C=1\)。

### 5.3 为什么简单模型反而更好

| 模型 | 单日 Accuracy | 单日 Macro-F1 |
|---|---:|---:|
| 改进 DeepLOB | 48.01% | 48.26% |
| 25维逻辑回归 | **60.63%** | **60.94%** |

完整交易日 walk-forward：

| 口径 | 训练日 → 测试日 | Accuracy | Macro-F1 |
|---|---|---:|---:|
| MBO | 07-09 → 07-21 | 58.85% | 59.46% |
| MBO | 07-09、07-21 → 08-07 | 58.48% | 58.04% |
| MBP | 07-21 → 07-22 | 63.35% | 63.33% |
| MBP | 07-21、07-22 → 08-07 | 58.85% | 54.41% |

### 5.4 当前训练集、验证集和测试集的具体区分

这里不是把所有 snapshot 随机切成训练/测试。每条数据链都按时间向前滚动，并且把“训练
分类器”和“校准/选择做市参数”分成两层：

| 数据链与目标测试日 | 分类器训练集 | 校准/策略选参验证集 | 最终测试集 |
|---|---|---|---|
| MBO → 07-21 | 07-09前70%，112,235个有效目标 | 07-09后30%，48,084个有效目标 | 07-21整日，79,045个有效目标 |
| MBO → 08-07 | 07-09、07-21整日，共239,483个有效目标 | 07-09后30% + 07-21整日样本外预测，共127,129个 | 08-07整日，61,560个 |
| MBP → 07-22 | 07-21前70%，55,225个有效目标 | 07-21后30%，23,652个有效目标 | 07-22整日，59,805个 |
| MBP → 08-07 | 07-21、07-22整日，共138,801个有效目标 | 07-21后30% + 07-22整日样本外预测，共83,457个 | 08-07整日，61,439个 |

“有效目标”已经剔除了100步输入历史不足、20步未来标签不足以及跨午休/交易时段的样本，
所以不等于原始快照行数。第一轮中，前70%只拟合分类器，后30%只做概率—tick校准和报价
参数选择。随后分类器可用完整历史日重训，再预测下一个完整未来日。

例如07-21在“预测07-21”这轮是完全未见测试日；到了“预测08-07”这轮，它已经是过去
信息，因此可以进入分类器训练集。同时，策略校准使用的是当时由07-09模型对07-21产生的
整日样本外预测，而不是在07-21上拟合后再预测07-21。这个角色转换是标准 walk-forward，
不构成向08-07泄漏。任何一轮的最终测试日都没有参与该轮的分类器拟合、波动率/成交强度
估计或 AS、GP-style、方向模型超参数选择。

逻辑回归胜出的数学原因不是线性模型天然优于深度学习，而是当前数据环境下更合适的
偏差—方差折中：

- 显式特征直接编码了与标签有关的价格路径和深度不平衡；
- 参数数量远小于 CNN–Inception–LSTM，估计方差较低；
- L2 正则使系数在共线、高度重叠样本中更稳定；
- 模型概率更容易检查和校准；
- 其限制也很明确：只能表示特征与 log-odds 的线性关系，复杂条件交互需要额外特征或非线性模型。

---

## 6. 主要误差来自标签含义与可执行涨幅不一致

先给出结论：本项目的三分类标签不是“在时刻 \(t\) 买入，到未来某个时刻卖出的涨跌”。它
预测的是**未来 \(k\) 个 snapshot 的平均中价，相对过去 \(k\) 个 snapshot 平均中价的变化**。
因此，即使逻辑回归准确预测了标签，也不等于准确预测了从当前时刻开始可以获得的交易收益。

### 6.1 标签包含一部分已经发生的价格变化

当前实现取 \(k=20\)，标签变量为：

$$
\bar m_t^-=\frac1k\sum_{i=0}^{k-1}m_{t-i},\qquad
\bar m_t^+=\frac1k\sum_{i=1}^{k}m_{t+i},
$$

$$
r_t^{label}=\frac{\bar m_t^+}{\bar m_t^-}-1.
$$

这个比率可以分解为：

$$
1+r_t^{label}
=\frac{\bar m_t^+}{\bar m_t^-}
=\underbrace{\frac{\bar m_t^+}{m_t}}_{\text{未来均价相对当前价}}
\underbrace{\frac{m_t}{\bar m_t^-}}_{\text{当前价相对过去均价}}.
$$

第二项 \(m_t/\bar m_t^-\) 描述的是在决策时刻之前已经发生的走势，它能够影响标签，却不能
成为时刻 \(t\) 之后的可交易利润。也就是说，模型可能只是识别到“当前价格已经高于过去
均价”，然后正确判断未来平均价格仍处于较高水平；这不代表从当前价格买入后还会继续上涨。

例如，过去20个 snapshot 的均价是100，当前中价已经涨到101，未来20个 snapshot 始终在
100.8附近。此时：

$$
r_t^{label}\approx\frac{100.8}{100}-1=+0.8\%,
$$

标签仍可能是上涨；但从当前101买入到未来100.8，实际是亏损：

$$
r_t^{current\rightarrow future}\approx\frac{100.8}{101}-1=-0.20\%.
$$

因此标签方向与当前入场后的收益方向可以直接相反。这是“60%左右分类准确率不能直接转化为
赚钱策略”的首要原因，而不只是手续费问题。

### 6.2 未来均值也不是实际退出价格

标签中的 \(\bar m_t^+\) 是未来20个 snapshot 的均值，它平滑了路径噪声，但交易不能以这20
个中价的平均值成交。真实主动交易更接近：预测后在下一快照以卖一价买入，之后在某一个明确
时刻以买一价卖出：

$$
R^{long}_{t,h}=\frac{b_{1,t+1+h}}{a_{1,t+1}}-1.
$$

做空对应：

$$
R^{short}_{t,h}=\frac{b_{1,t+1}}{a_{1,t+1+h}}-1.
$$

它与 \(r_t^{label}\) 至少有三层差异：

1. **起点不同**：标签从过去均价出发，交易从当前之后的可执行买一/卖一价出发；
2. **终点不同**：标签使用一段未来路径的均值，交易只能选择某个具体退出时刻和价格；
3. **时间尺度不同**：20个 snapshot 是事件时间，不是固定20秒；行情活跃时很短，清淡时很长。

因此，从标签概率映射到实际持有期收益时会产生较大的**目标错配误差**。

### 6.3 直接交易实验：分类准确率没有转化为正收益

为直接检验“逻辑回归预测上涨就买、预测下跌就卖”能否赚钱，使用不依赖07-09数据的 MBP
walk-forward 链做主动吃单实验：

- 预测上涨：在下一快照以卖一价买入，20个 snapshot 后以买一价卖出；
- 预测下跌：在下一快照以买一价卖出，20个 snapshot 后以卖一价买回；
- 预测平稳：不交易；
- 持仓互不重叠，买卖价差已经包含在毛收益中；
- `官方净收益`再扣每次完整往返2.54bp固定费用；`含佣金净收益`共扣5.54bp。

结果如下。Accuracy 和 Macro-F1 衡量原始三分类标签；收益衡量从可执行买一/卖一价出发的
真实持有期方向：

| 训练日 → 测试日 | Accuracy | Macro-F1 | 交易数 | 价差后毛收益/笔 | 官方净收益/笔 | 含佣金净收益/笔 | 毛胜率 | 毛 Profit Factor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MBP 07-21 → 07-22 | **63.35%** | **63.33%** | 2,338 | **−5.18bp** | −7.72bp | −10.72bp | 23.99% | 0.293 |
| MBP 07-21、07-22 → 08-07 | **58.85%** | **54.41%** | 2,703 | **−7.98bp** | −10.52bp | −13.52bp | 15.32% | 0.171 |

这里最重要的不是扣费后亏损，而是**在手续费设为零、只保留真实买卖价差时，平均每笔已经
分别亏损5.18bp和7.98bp**。Profit Factor 分别只有0.293和0.171，远低于盈亏平衡所需的1。
提高置信度门槛也没有解决问题：07-22在0.5门槛下仍为−5.04bp/笔，08-07仍为−8.22bp/笔。

因此实验支持的是：逻辑回归确实能预测原标签，但该标签的正确方向并不等于从当前可执行
价格进入后的正确收益方向。亏损不是单纯由手续费造成的，标签—交易目标错配和买卖价差在
扣除额外费用之前已经使策略期望为负。完整机器结果见
[`output/multiday_strategy_results.json`](output/multiday_strategy_results.json)。

### 6.4 逻辑回归优化的也不是交易目标

逻辑回归最小化的是分类损失：

$$
\mathcal L_{LR}
=-\sum_i\omega_{y_i}\log p(y_i\mid X_i)
+\frac{1}{2C}\lVert W\rVert_F^2,
$$

而交易真正关心的是从可执行价格出发、扣除成本后的条件期望：

$$
\mathbb E[R_{t,h}^{executable}\mid X_t]
-\text{fees}-\text{slippage}-\text{impact}.
$$

分类损失既没有惩罚“标签预测正确但当前入场亏损”，也没有区分0.5bp和10bp的价格变化。
所以逻辑回归在这里能够证明订单簿对**平滑后的相对方向标签**含有信息，却不能单独证明存在
可执行交易 alpha。

### 6.5 在做市模型中应把它视为弱附加信号

这并不意味着方向概率没有价值。可以先定义：

$$
g_t=p_t(up)-p_t(down),
$$

但不能直接把 \(g_t\) 当作未来收益或漂移。必须在更早的验证区间内，用“当前时刻之后”的
实际中价变化、execution markout 或 adverse-selection cost 对它重新校准，再用于报价中心、
单侧成交过滤和库存管理。

因此本项目更准确的经济命题是：**逻辑回归可能识别与未来局部价格状态有关的信息，但由于
原标签不是当前到未来的可执行涨幅，该信息只能作为经过重新校准的做市附加状态，不能直接
视为赚钱策略。**

---

## 7. 第三阶段：把方向信号加入 AS/GP 做市

### 7.1 经典 AS 的数学结构

做市商在参考价 \(S_t\) 两侧挂买卖单：

$$
p_t^b=S_t-\delta_t^b,\qquad
p_t^a=S_t+\delta_t^a.
$$

限价单成交强度常写成随报价距离指数衰减：

$$
\lambda^b(\delta)=A e^{-k\delta},\qquad
\lambda^a(\delta)=A e^{-k\delta}.
$$

若买单成交，库存和现金变化为：

$$
dq_t=+1,qquad dX_t=-p_t^b;
$$

若卖单成交：

$$
dq_t=-1,qquad dX_t=+p_t^a.
$$

经典 AS 用指数效用平衡点差收益和库存风险：

$$
\max_{\delta^a,\delta^b}
\mathbb E\left[-\exp\left(-\gamma(X_T+q_TS_T)\right)\right],
$$

其中 \(\gamma>0\) 是风险厌恶系数。零漂移布朗运动下的保留价近似为：

$$
r_t=S_t-q_t\gamma\sigma^2(T-t).
$$

库存为正时，保留价下降，使卖价更积极、买价更保守；库存为负时反向。经典指数强度近似下，
总点差包含库存风险项和流动性项：

$$
\delta_t^a+\delta_t^b
\approx \gamma\sigma^2(T-t)
+\frac{2}{\gamma}\log\left(1+\frac{\gamma}{k}\right).
$$

GP 模型进一步强调离散 tick、随机价差状态、限价单优先级以及必要时用市价单主动降低库存。
我们的实现是面向真实逐笔数据的 **AS/GP-style 离散近似**，不是原论文 HJB/QVI 的逐状态
数值全解。

为避免把“方向信号消融”误当成“传统模型基准”，最终实验把四个策略分开：

1. **经典 AS**：按上式计算保留价和总点差；
2. **GP-style 离散基准**：无方向信号，使用离散 tick、库存偏移、硬库存上限和日末平仓；
3. **方向+库存**：在 GP-style 离散控制上加入校准后的方向漂移；
4. **同参数无信号消融**：仅把第3项的 \(\eta\) 置零，用来识别方向信号的边际贡献。

经典 AS 的 \(\sigma_h^2\) 用验证期一个20快照报价生命周期的中价变化方差估计，\(k\) 用
验证期0/1/2 tick 距离下的 through 成交率做 \(\log P(fill)\) 对距离的线性拟合。三个主模型
都只在同一历史验证期独立选参，统一最大化

$$
J=PnL_{net}-0.25\,MaxDrawdown.
$$

测试日不参与波动率、成交强度或超参数选择。GP 基准仍应称为 **GP-style**：若要声称精确
复现 Guilbaud–Pham，必须进一步估计随机价差转移和队列状态并数值求解原论文 QVI。

### 7.2 方向概率变成条件漂移

逻辑回归输出：

$$
u_t=p_t(\mathrm{up}),\qquad
d_t=p_t(\mathrm{down}).
$$

方向分数定义为：

$$
g_t=u_t-d_t\in[-1,1].
$$

它比硬分类保留更多置信度信息。用更早验证期将方向分数线性校准到未来中价 tick：

$$
\widehat{\Delta m}_t^{tick}
=\operatorname{clip}(\beta_0+\beta_1g_t,-2,2),
$$

$$
\beta_1=max\left(
0,
\frac{\operatorname{Cov}(g_t,\Delta m_t^{tick})}
{\operatorname{Var}(g_t)}
\right).
$$

如果验证期相关为负，斜率截为零，而不是事后把信号方向反转。

### 7.3 方向和库存共同移动报价中心

设库存为 \(q_t\)，硬上限为 \(q_{max}\)。实际离散中心偏移（单位 tick）为：

$$
c_t=eta\widehat{\Delta m}_t^{tick}
-\kappa\frac{q_t}{q_{max}},
$$

其中 \(\eta\) 控制方向强度，\(\kappa\) 控制库存惩罚。给定基础距离 \(d_0\)：

$$
\tilde p_t^b=b_{1,t}-d_0\tau+c_t\tau,
$$

$$
\tilde p_t^a=a_{1,t}+d_0\tau+c_t\tau,
$$

其中 \(\tau=0.02\) HKD 是 tick。取整并强制被动：

$$
p_t^b\le a_{1,t}-\tau,
\qquad
p_t^a\ge b_{1,t}+\tau,
\qquad
p_t^b<p_t^a.
$$

直观上：

- \(c_t>0\)：买价提高、卖价提高；买单更积极，卖单离市场更远；
- \(c_t<0\)：买价降低、卖价降低；卖单更积极，买单更远；
- \(q_t>0\)：库存项使中心下移，鼓励卖出并抑制继续买入；
- \(|q_t|=q_{max}\)：停止继续增加同方向库存。

---

## 8. 成交、现金、费用与 PnL

每次 maker fill 交易一手 \(L=100\) 股。买单成交：

$$
q\leftarrow q+1,qquad
X\leftarrow X-Lp^b.
$$

卖单成交：

$$
q\leftarrow q-1,qquad
X\leftarrow X+Lp^a.
$$

若按成交金额比例收取每边费率 \(f\)：

$$
Fee=\sum_j Lp_j f.
$$

测试结束将剩余库存按一档价平仓，最终净 PnL：

$$
PnL_{net}=X_T+Lq_Tp_T^{flatten}-Fee.
$$

当前报告三种经济口径：零费用毛收益、仅官方固定费用、固定费用加假设券商佣金。主表的
all-in 情景使用每边2.77bp。

由于缺少虚拟订单的真实队列位置，成交采用两个敏感性边界：

- **touch**：后续成交价到达挂单价即认为成交，偏乐观；
- **through**：后续成交价至少穿过挂单一档才认为成交，偏保守。

同一报价区间每边最多成交一次，并加入5ms假设延迟。生产级回测仍需要 queue-ahead、订单
确认、撤单确认和真实 maker 费率。

---

## 9. Markout 与逆向选择

设 maker 方向符号：买入 \(z_j=+1\)，卖出 \(z_j=-1\)。成交价为 \(p_j\)，成交后即时中价
为 \(m_j\)，\(h\) 秒后的中价为 \(m_{j,h}\)。

### 9.1 Execution markout

$$
M_j(h)=z_j\frac{m_{j,h}-p_j}{p_j}10^4.
$$

它从真实成交价出发，因此包含点差收益。\(M_j(h)>0\) 表示这笔成交在 \(h\) 秒后对做市商
仍有正价值。

### 9.2 逆向选择成本

$$
ASCost_j(h)=-z_j\frac{m_{j,h}-m_j}{m_j}10^4.
$$

买入后中价下跌或卖出后中价上涨会得到正成本；数值越低越好。方向增强相对无信号的成本
降低定义为：

$$
\Delta ASCost(h)=
\overline{ASCost}_{no\ signal}(h)
-\overline{ASCost}_{directional}(h).
$$

\(\Delta ASCost>0\) 才表示方向增强改善。

### 9.3 配对增量 PnL

方向与无信号策略使用相同基础距离、库存惩罚、库存上限、撤挂周期、日期和费用，仅令对照组
\(\eta=0\)：

$$
\Delta PnL=PnL_{directional}-PnL_{\eta=0}.
$$

这是策略级配对；方向不同会导致库存路径和成交集合不同，不能解释为同一笔订单逐笔配对。

---

## 10. AS、GP-style 与方向+库存的公平对比

全部 PnL 均含每边2.77bp费用、相同5ms延迟、20快照撤挂周期和日末一档平仓。`through`
是保守成交假设，`touch` 是乐观成交假设；它们是成交敏感性情景，不是统计置信区间，也不
保证构成严格上下界。

| 数据口径 | 测试日 | 情景 | 方向+库存 | 经典 AS | GP-style | 方向−AS | 方向−GP |
|---|---|---|---:|---:|---:|---:|---:|
| MBO | 07-21 | through | −1,164.54 | **−614.33** | −3,254.49 | −550.21 | **+2,089.95** |
| MBO | 07-21 | touch | −235.15 | **+516.16** | −2,175.32 | −751.31 | **+1,940.17** |
| MBO | 08-07 | through | **+622.22** | −210.25 | −572.63 | **+832.46** | **+1,194.85** |
| MBO | 08-07 | touch | **+1,448.72** | +367.08 | +417.06 | **+1,081.65** | **+1,031.67** |
| MBP | 07-22 | through | −5,526.06 | −5,551.15 | **−5,289.69** | +25.09 | −236.37 |
| MBP | 07-22 | touch | **−3,048.53** | −3,414.33 | −3,275.55 | +365.80 | +227.01 |
| MBP | 08-07 | through | −1,439.81 | **−1,049.88** | −1,095.21 | −389.94 | −344.60 |
| MBP | 08-07 | touch | −1,047.39 | −311.97 | **−238.07** | −735.42 | −809.32 |

结果不支持“方向模型稳定优于经典模型”。MBO 两个测试日合计，方向+库存在 through 下为
−542.32 HKD，优于 AS 的−824.58和 GP-style 的−3,827.12；touch 下为+1,213.57，也优于
AS 的+883.24和 GP-style 的−1,758.26。但 MBP 两日合计恰好相反：方向+库存在 through
下为−6,965.87，差于 AS 的−6,601.03和 GP-style 的−6,384.90；touch 下也最差。

因此当前最窄且可靠的结论是：**方向信号在 MBO 链中提高了模型级结果，但该优势没有跨到
MBP 链；只有 MBO 08-07 的方向+库存在保守 through 下实现正净 PnL。**

同参数无信号消融仍单独保留。保守 through 下，方向信号相对消融组的增量依次为：MBO
07-21 `+2,329.77`、MBO 08-07 `+1,643.76`、MBP 07-22 `−236.37`、MBP 08-07
`−74.52` HKD。这个表只回答“方向项本身的边际价值”，不能替代上面的模型级基准。

20秒买卖侧逆向选择也呈混合结果：

| 口径/测试日 | 买单成本降低 | 卖单成本降低 |
|---|---:|---:|
| MBO 07-21 | +0.439bp | −0.170bp |
| MBO 08-07 | +0.435bp | +3.855bp |
| MBP 07-22 | +0.633bp | +0.233bp |
| MBP 08-07 | +3.182bp | −1.077bp |

方向增强没有稳定改善每一侧、每个 horizon 的平均 markout。MBO 07-21 的增量主要来自把
through 成交从2,778次降到1,467次，减少低质量成交和费用；MBO 08-07的增量则主要来自
毛 PnL 改善。当前信号更接近**动态报价与成交频率过滤器**，而非稳定的逐笔方向 alpha。

---

## 11. 项目中使用的核心数学原则

### 11.1 条件期望而不是硬分类

做市决策需要的是条件经济价值：

$$
\mathbb E[\Delta m\mid X_t],
$$

而不仅是 \(\arg\max_k p(y=k\mid X_t)\)。概率差和校准步骤是在把分类问题转回条件期望问题。

### 11.2 偏差—方差折中

DeepLOB 表达能力强、偏差低，但在少量独立交易日上方差高；显式特征逻辑回归表达受限，却
因参数少、正则强而具有更好的跨时段稳定性。当前结果体现的是数据规模下的最优复杂度，不是
“深度学习无效”。

### 11.3 非独立同分布与有效样本量

100步滑动窗口高度重叠，\(N\) 个窗口并不等于 \(N\) 个独立样本。市场状态还随日期、时段、
波动率和价差改变，违反 i.i.d. 假设。因此不能随机打乱窗口后用普通测试集准确率代表实盘
泛化；必须按时间和完整交易日 walk-forward。

### 11.4 概率校准

分类概率不是价格变化单位。校准将 \(p(up)-p(down)\) 映射为 tick，使信号强度能与 tick、
库存惩罚和点差放在同一量纲中。校准相关很低时，即使 Accuracy 较高，也说明经济幅度弱。

### 11.5 随机控制与库存风险

AS/GP 的核心不是预测，而是在点差收益、成交概率和库存风险之间动态权衡。方向信号只改变
参考中心，库存惩罚和硬上限仍负责防止模型把预测置信度变成无限方向仓位。

### 11.6 逆向选择

被动成交不是随机样本。对手愿意打掉买单或卖单，往往意味着价格即将向做市商不利方向移动。
所以 maker 策略必须评价“成交条件下”的未来中价：

$$
\mathbb E[\Delta m\mid X_t,\text{our quote filled}],
$$

而不是无条件方向准确率。

### 11.7 交易成本使统计优势与经济优势分离

一个预测模型可以统计显著，却没有经济收益。真正的报价价值必须满足：

$$
\text{spread capture}
+\text{conditional markout}
-\text{fees}
-\text{inventory risk}
-\text{impact}
>0.
$$

---

## 12. 下一阶段的统一数学目标

当前分类、成交和 markout 是分阶段估计。更完整的方向增强做市模型应直接估计每一侧的条件
净边际价值。

对候选买价 \(p_t^b\)：

$$
Edge_t^b=
P(F_t^b=1\mid X_t,p_t^b)
\left[
\mathbb E[m_{t+h}-p_t^b\mid X_t,F_t^b=1]
-C_t^b
\right],
$$

对候选卖价 \(p_t^a\)：

$$
Edge_t^a=
P(F_t^a=1\mid X_t,p_t^a)
\left[
\mathbb E[p_t^a-m_{t+h}\mid X_t,F_t^a=1]
-C_t^a
\right].
$$

其中：

- \(F_t^b,F_t^a\) 是各侧是否成交；
- 条件期望项是各侧 execution markout；
- \(C_t^s\) 包含费用、延迟、冲击和库存风险；
- 只有 \(Edge_t^s>0\) 时才应在该侧报价。

加入库存风险后，可优化：

$$
\max_{p_t^a,p_t^b}
Edge_t^a+Edge_t^b
-\lambda_q q_t^2
-\lambda_{dd}\,\text{DrawdownRisk}_t.
$$

这个目标把本项目的三条线统一起来：方向预测提供条件漂移，成交模型提供 fill probability，
markout 提供逆向选择价值，AS/GP 提供库存和动态控制框架。

---

## 13. 证据边界、稳健性与失败模式

1. 完整、严格且同口径的 MBO 日只有07-09、07-21、08-07，不构成连续多周样本。
2. 只有07-21和07-22相邻，且两者同时可用时只能走 MBP 口径。
3. 真实 queue position 不可观测；touch/through 是成交敏感性边界，不是精确交易所回放。
4. MBO 与 MBP 结果不一致，可能来自恢复口径、成交定义或市场状态差异，当前不能选择性只报告
   MBO 的正结果。
5. 单日60.63%的改进实验已经在迭代中看过同一后30% holdout，不能再称为完全未见测试。
6. 方向与无信号策略成交集合不同，平均 markout 差异同时包含报价选择和样本构成变化。
7. 当前未包含真实券商最低佣金、maker rebate、订单确认/撤单延迟、市场冲击和真实排队数量。

独立验证状态：

- 盘口重建和切分单元测试14项通过；
- 方向做市成交日志16,565条的10项核算检查通过；
- 5/10/20秒 markout 记录115,227条的13项因果、公式、口径、配对和模型差值检查通过；
- 当前整体结论为 **Share with caveats**，不应直接实盘。

---

## 14. 推荐的下一步实验设计

1. 收集至少10–20个连续交易日的同一 MBO 格式，最好包含交易所序列号和周期全量快照。
2. 固定滚动协议，例如过去5日拟合与校准、第6日验证参数、第7日只测试；最终保留完整冻结周。
3. 把快照 horizon 改为真实时间 horizon，统一使用1/5/10/20秒标签和撤挂周期。
4. 分别训练买单和卖单的 fill probability 与条件 markout，不再强制两侧共享同一个方向效果。
5. 将 quote distance、spread state、queue imbalance、time-of-day 和 volatility 加入成交强度模型。
6. 若取得 queue-ahead 数据，使用生存分析或 hazard model 估计排队成交概率。
7. 以日级 PnL、最大回撤、库存分布和 paired incremental PnL 作为主指标；Accuracy 只作为诊断。
8. 使用日级或区块 bootstrap，而不是把高度重叠窗口当成独立样本计算显著性。

进一步需要回答的问题：

- MBO/MBP 差异主要来自盘口恢复、成交方向标记还是抽样频率？
- 方向信号是否只在某些价差、波动率或时段状态有效？
- MBO 07-21的收益改善能否在更多日期复现，还是仅仅减少了当日换手？
- DeepLOB 在获得足够连续日后，能否在条件 markout 或 fill prediction 上超过逻辑回归？

---

## 15. 项目文件与复现入口

### 数据与盘口重建

- [`lob_reconstruction.py`](lob_reconstruction.py)：MBO/MBP 严格重放。
- [`tests/test_reconstruction.py`](tests/test_reconstruction.py)：重建、切分和因果特征测试。

### DeepLOB 与逻辑回归

- [`run_experiment.py`](run_experiment.py)：初始单日 DeepLOB。
- [`run_improved_experiment.py`](run_improved_experiment.py)：正则化 DeepLOB、25维因果逻辑回归。
- [`run_multiday_strategy.py`](run_multiday_strategy.py)：跨日分类和直接交易检验。
- [`multiday_strategy_analysis.ipynb`](multiday_strategy_analysis.ipynb)：已执行多日 notebook。

### 方向增强做市与 markout

- [`directional_market_maker.py`](directional_market_maker.py)：信号校准、离散报价、库存和成交核心。
- [`run_directional_market_maker.py`](run_directional_market_maker.py)：AS/GP-style walk-forward。
- [`run_directional_markout.py`](run_directional_markout.py)：5/10/20秒双侧 markout 与配对增量 PnL。
- [`directional_markout_analysis.ipynb`](directional_markout_analysis.ipynb)：已执行结果 notebook。
- [`output/directional_markout_results.json`](output/directional_markout_results.json)：机器可读结果。
- [`output/directional_markout_validation.json`](output/directional_markout_validation.json)：独立复核。

### 复现命令

```powershell
python -m unittest discover -s 0824/tests -v
python 0824/run_experiment.py --device cuda
python 0824/run_improved_experiment.py
python 0824/run_multiday_strategy.py
python 0824/run_directional_market_maker.py
python 0824/run_directional_markout.py
python 0824/validate_directional_markout.py
```

更详细的逐阶段结果、费用来源和异常统计见
[`reconstruction_and_results.md`](reconstruction_and_results.md)。

---

## 参考模型

- Zhang, Zohren and Roberts, *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books*.
- Avellaneda and Stoikov, [*High-frequency trading in a limit order book*](https://math.nyu.edu/inmemoriam/avellaneda/HighFrequencyTrading.pdf).
- Guilbaud and Pham, [*Optimal High Frequency Trading with limit and market orders*](https://arxiv.org/abs/1106.5040).
- Guilbaud and Pham, [*Optimal High Frequency Trading in a Pro-Rata Microstructure with Predictive Information*](https://arxiv.org/abs/1205.3051).
