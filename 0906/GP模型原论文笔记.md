# Guilbaud–Pham（GP）高频做市模型笔记

_按“校准模型 → 定义目标 → 定义价值函数 → 推导 QVI → 离线求解 → 在线报价”的逻辑梳理 GP 原论文。_

---

## 📋 主线结论

GP 模型不是从 LOB 直接得到一条固定报价公式。它先用历史盘口与成交数据估计市场状态转移和限价单成交强度，再给定收益—风险目标，由动态规划得到价值函数满足的 HJB-QVI，最后从价值函数的最优控制项中读出报价、挂单量和市价单动作。[^1]

```mermaid
flowchart LR
    accTitle: GP 模型完整求解主线
    accDescr: 历史盘口与成交数据先校准外生市场模型，再由目标函数定义价值函数并离线求解 QVI，最后在线按当前状态查询最优动作。

    data([📊 历史 LOB 数据]) --> calibrate[🔍 校准模型参数]
    calibrate --> objective[🎯 给定优化目标]
    objective --> value[📈 定义价值函数]
    value --> qvi[⚙️ 推导并求解 QVI]
    qvi --> policy[(📋 最优策略表)]
    state([📥 在线当前状态]) --> lookup[🔍 查询策略表]
    policy --> lookup
    lookup --> action([✅ 报价与交易动作])

    classDef input_style fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#3b0764
    classDef process_style fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef result_style fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class data,state input_style
    class calibrate,objective,value,qvi,lookup process_style
    class policy,action result_style
```

最重要的区分是：

- 历史数据用于估计外生参数，不是直接回归价值函数
- 价值函数由优化目标定义，并通过 HJB-QVI 求出
- 最优报价是 QVI 中 `argmax` 对应的动作
- 论文的数值特例中，策略可离线整理成 `时间 × 库存 × spread 状态 → 最优动作` 的查找表

## 🔤 符号与状态

| 符号 | 定义 | 取值或单位 |
| --- | --- | --- |
| $t,T$ | 当前时间与交易终点 | $t\in[0,T]$ |
| $X_t$ | 现金余额 | 货币 |
| $Y_t$ | 库存；正数为多头 | 股数 |
| $P_t$ | 中间价 | 货币 / 股 |
| $S_t$ | best ask 减 best bid | $\{\delta,2\delta,\ldots,m\delta\}$ |
| $\delta$ | tick size | 货币 / 股 |
| $i$ | spread 状态编号 | $S_t=i\delta$ |
| $q^b,q^a$ | bid、ask 报价档位 | $Bb,Bb_+,Ba,Ba_-$ |
| $\ell^b,\ell^a$ | bid、ask 限价单量 | $[0,\bar\ell]$ |
| $e$ | 市价单量；正为买、负为卖 | $[-\bar e,\bar e]$ |
| $\lambda_i^b,\lambda_i^a$ | 限价买、卖单成交强度 | 次 / 时间 |
| $\alpha$ | 从当前到 $T$ 的完整策略 | 限价单控制与市价单控制 |
| $v$ | 完整状态价值函数 | $v(t,x,y,p,s)$ |
| $v_i$ | 固定 spread 状态后的价值函数 | $v_i(t,x,y,p)=v(t,x,y,p,i\delta)$ |
| $\phi_i$ | 鞅价格、线性效用下的附加做市价值 | $v_i=x+yp+\phi_i(t,y)$ |

## 🔍 第一步：从 LOB 数据校准市场模型

### 中间价过程

一般模型只假设 $P_t$ 是外生 Markov 过程，状态空间为 $\mathbb P$，无穷小生成元为 $\mathcal P$。论文举例包括 Lévy 过程、算术布朗运动和指数 Lévy 过程，但一般模型没有固定为某一个 SDE。[^1]

因此，一般情况下不能直接写成

$$
\mathcal Pv=\mu\partial_pv+\frac12\sigma^2\partial_{pp}v,
$$

除非另外指定

$$
dP_t=\mu\,dt+\sigma\,dW_t.
$$

论文用于主要数值实验的特例只要求 $P_t$ 是鞅，即

$$
\mathcal Pp=0.
$$

在算术布朗运动下，这等价于 $\mu=0$；对一般带跳过程，它表示包含跳跃补偿在内的总漂移为零。

### 离散 spread 过程

spread 只能取有限个 tick 倍数：

$$
S_t\in\mathbb S=\delta\mathbb I_m,
\qquad
\mathbb I_m=\{1,\ldots,m\}.
$$

GP 先定义 tick-time 下的离散 Markov 链 $\hat S_n$：

$$
\Pr(\hat S_{n+1}=j\delta\mid \hat S_n=i\delta)=\rho_{ij},
\qquad \rho_{ii}=0,
$$

再用强度为确定性函数 $\lambda(t)$ 的 Poisson 时钟 $N_t$ 将其变换到日历时间：

$$
S_t=\hat S_{N_t}.
$$

于是 $S_t$ 是时变连续时间 Markov 链，其强度矩阵为

$$
r_{ij}(t)=\lambda(t)\rho_{ij},\quad i\ne j,
\qquad
r_{ii}(t)=-\sum_{j\ne i}r_{ij}(t).
$$

论文假设 $P$ 与 $S$ 相互独立。固定 $s=i\delta$ 后，spread 生成元为

$$
R(t)v_i
=
\sum_{j=1}^{m}r_{ij}(t)
\left[v_j(t,x,y,p)-v_i(t,x,y,p)\right].
$$

历史数据在这里用于估计 $\rho_{ij}$ 和 $\lambda(t)$，从而得到 $r_{ij}(t)$。

### 可选报价与成交价格

市场 best bid 和 best ask 为

$$
P_t^b=P_t-\frac{S_t}{2},
\qquad
P_t^a=P_t+\frac{S_t}{2}.
$$

当 $s>\delta$ 时，做市商可以挂在当前最优价，也可以改善一个 tick：

$$
\mathcal Q^b=\{Bb,Bb_+\},
\qquad
\mathcal Q^a=\{Ba,Ba_-\}.
$$

实际成交报价为

$$
\pi^b(q^b,p,s)=
\begin{cases}
p-\dfrac{s}{2},&q^b=Bb,\\
p-\dfrac{s}{2}+\delta,&q^b=Bb_+,
\end{cases}
$$

$$
\pi^a(q^a,p,s)=
\begin{cases}
p+\dfrac{s}{2},&q^a=Ba,\\
p+\dfrac{s}{2}-\delta,&q^a=Ba_-.
\end{cases}
$$

若 $s=\delta$，改善一个 tick 会跨过全部点差，等价于主动成交，因此被动报价集合退化为

$$
\mathcal Q_1^b=\{Bb\},
\qquad
\mathcal Q_1^a=\{Ba\}.
$$

### 限价单成交强度

买、卖侧成交事件分别由独立 Cox 过程 $N^b,N^a$ 描述，条件强度为

$$
\lambda^b(q^b,s),
\qquad
\lambda^a(q^a,s).
$$

改善一个 tick 应带来更高成交强度：

$$
\lambda^b(Bb,s)<\lambda^b(Bb_+,s),
\qquad
\lambda^a(Ba,s)<\lambda^a(Ba_-,s).
$$

论文假设对手市价单到达时会完全成交本方这笔限价单。因此限价单成交导致

$$
dY_t=L_t^b\,dN_t^b-L_t^a\,dN_t^a,
$$

$$
dX_t=-\pi^b(Q_t^b,P_{t-},S_{t-})L_t^b\,dN_t^b
+\pi^a(Q_t^a,P_{t-},S_{t-})L_t^a\,dN_t^a.
$$

所以仅估计 $P,S$ 的动态还不够；$\lambda^b,\lambda^a$ 是决定最优报价的关键输入。

### 市价单成本

发送大小为 $e$ 的市价单时，$e>0$ 表示买入，$e<0$ 表示卖出。论文的基础成本函数为

$$
c(e,p,s)=ep+|e|\frac{s}{2}+\varepsilon,
$$

其中 $\varepsilon>0$ 是固定费用。交易后的状态跳变为

$$
(x,y)\longrightarrow
\bigl(x-c(e,p,s),y+e\bigr).
$$

## 🎯 第二步：先定义什么叫“最优”

模型参数只描述市场怎样随机变化，本身不能回答应该怎样报价。要定义“最优”，还必须给出目标函数：

$$
\sup_{\alpha\in\mathcal A}
\mathbb E_{t,x,y,p,s}
\left[
U\bigl(L(X_T,Y_T,P_T,S_T)\bigr)
-\gamma\int_t^Tg(Y_u)\,du
\right].
$$

其中 $\alpha=(\alpha^{make},\alpha^{take})$ 包含从当前到 $T$ 的全部决策：连续更新的限价单档位和数量，以及离散时点发送的市价单。

若期末立即用市价单清掉库存 $y$，清算财富为

$$
L(x,y,p,s)
=x-c(-y,p,s)
=x+yp-|y|\frac{s}{2}-\varepsilon.
$$

目标中的两部分承担不同角色：

| 项 | 作用 | 进入动态规划的位置 |
| --- | --- | --- |
| $U(L(X_T,Y_T,P_T,S_T))$ | 衡量期末清算财富 | 终端条件 |
| $-\gamma\int_t^Tg(Y_u)du$ | 惩罚持仓期间的库存风险 | HJB 中的运行成本 |

因此，目标函数是“价值函数 → QVI → 最优报价”整条链条的起点。更换 $U$、$g$ 或 $\gamma$，即使市场分布完全相同，也会得到不同的价值函数和最优策略。

## 📈 第三步：由目标函数定义价值函数

价值函数就是：当前状态给定时，未来采用最优策略所能得到的最大条件期望：

$$
\boxed{
v(t,x,y,p,s)
=
\sup_{\alpha\in\mathcal A}
\mathbb E_{t,x,y,p,s}
\left[
U\bigl(L(X_T,Y_T,P_T,S_T)\bigr)
-\gamma\int_t^Tg(Y_u)\,du
\right].
}
$$

它的终端条件直接来自清算目标：

$$
v(T,x,y,p,s)=U(L(x,y,p,s)).
$$

由于 spread 只有 $m$ 个状态，论文定义

$$
v_i(t,x,y,p):=v(t,x,y,p,i\delta),
\qquad i=1,\ldots,m.
$$

$v$ 是包含离散变量 $s$ 的紧凑写法；$(v_1,\ldots,v_m)$ 是把每个 spread 状态展开后的向量写法。两者表示同一个对象。

## ⚙️ 第四步：动态规划给出 HJB-QVI

### 限价单生成元

对候选限价单控制 $q=(q^b,q^a)$、$\ell=(\ell^b,\ell^a)$，论文定义

$$
\begin{aligned}
\mathcal L^{q,\ell}v
={}&\mathcal Pv+R(t)v\\
&+\lambda^b(q^b,s)
\left[
v(t,x-\pi^b\ell^b,y+\ell^b,p,s)-v(t,x,y,p,s)
\right]\\
&+\lambda^a(q^a,s)
\left[
v(t,x+\pi^a\ell^a,y-\ell^a,p,s)-v(t,x,y,p,s)
\right],
\end{aligned}
$$

其中 $\pi^b=\pi^b(q^b,p,s)$、$\pi^a=\pi^a(q^a,p,s)$。两个成交项都具有

$$
\text{成交强度}\times\text{一次成交后的价值变化}
$$

的结构。

### 市价单干预算子

立即执行最优市价单后的价值为

$$
\mathcal Mv(t,x,y,p,s)
=
\sup_{e\in[-\bar e,\bar e]}
v(t,x-c(e,p,s),y+e,p,s).
$$

由于立即发市价单始终是可选动作，价值函数必须满足

$$
v\geq\mathcal Mv.
$$

### 完整 QVI

GP 的动态规划方程为

$$
\boxed{
\min\left[
-\partial_tv
-\sup_{(q,\ell)\in\mathcal Q(s)\times[0,\bar\ell]^2}
\mathcal L^{q,\ell}v
+\gamma g(y),
\quad
v-\mathcal Mv
\right]=0.
}
$$

外层 `min` 不对某个变量求最小值，而是在比较两个 QVI 残差。真正的控制优化发生在 $\sup_{q,\ell}$ 和 $\sup_e$ 中。令

$$
A=-\partial_tv-\sup_{q,\ell}\mathcal L^{q,\ell}v+\gamma g(y),
\qquad
B=v-\mathcal Mv,
$$

则 $\min(A,B)=0$ 等价于 $A\geq0$、$B\geq0$，并且至少有一个等于零：

| 区域 | 条件 | 最优动作 |
| --- | --- | --- |
| 继续区域 | $v>\mathcal Mv$，因此 $A=0$ | 不发市价单；选择最优限价单 |
| 干预区域 | $v=\mathcal Mv$ | 立即执行达到 $\mathcal Mv$ 的市价单 |

严格说，这不是单一的普通 PDE，而是包含离散 spread 跳转、成交跳变和脉冲控制的耦合非局部 QVI 系统。

## 🧮 第五步：在鞅价格特例中降维并离线求解

论文主要数值实验采用

$$
U(w)=w,
\qquad P_t\text{ 是鞅}.
$$

此时

$$
\boxed{
v_i(t,x,y,p)=x+yp+\phi_i(t,y).
}
$$

三部分分别表示当前现金、按中间价计价的库存，以及在 spread 状态 $i$ 下未来继续交易的附加价值。由于 $\mathcal Pp=0$，现金 $x$ 和绝对价格 $p$ 可从控制问题中消去，真正需要求解的是 $m$ 个相互耦合的函数 $\phi_i(t,y)$。

例如，bid 限价单成交后的价值增量化为

$$
\phi_i(t,y+\ell^b)-\phi_i(t,y)
+\left(\frac{i\delta}{2}-\delta\mathbf 1_{\{q^b=Bb_+\}}\right)\ell^b,
$$

ask 侧则为

$$
\phi_i(t,y-\ell^a)-\phi_i(t,y)
+\left(\frac{i\delta}{2}-\delta\mathbf 1_{\{q^a=Ba_-\}}\right)\ell^a.
$$

降维后的 QVI 为

$$
\begin{aligned}
\min\Bigg[&
-\partial_t\phi_i
-\sum_{j=1}^{m}r_{ij}(t)\bigl[\phi_j(t,y)-\phi_i(t,y)\bigr]\\
&-\sup_{(q^b,\ell^b)\in\mathcal Q_i^b\times[0,\bar\ell]}
\lambda_i^b(q^b)
\left[
\phi_i(t,y+\ell^b)-\phi_i(t,y)
+\left(\frac{i\delta}{2}-\delta\mathbf 1_{\{q^b=Bb_+\}}\right)\ell^b
\right]\\
&-\sup_{(q^a,\ell^a)\in\mathcal Q_i^a\times[0,\bar\ell]}
\lambda_i^a(q^a)
\left[
\phi_i(t,y-\ell^a)-\phi_i(t,y)
+\left(\frac{i\delta}{2}-\delta\mathbf 1_{\{q^a=Ba_-\}}\right)\ell^a
\right]
+\gamma g(y),\\
&\phi_i(t,y)
-\sup_{e\in[-\bar e,\bar e]}
\left[
\phi_i(t,y+e)-\frac{i\delta}{2}|e|-\varepsilon
\right]
\Bigg]=0,
\end{aligned}
$$

终端条件为

$$
\phi_i(T,y)=-|y|\frac{i\delta}{2}-\varepsilon.
$$

数值上，从终端条件开始沿时间反向递推，并在离散库存 $y$ 和 spread 状态 $i$ 上联立求解。原论文使用有限差分离散时间和库存空间；每个网格点同时保存最优动作。[^1]

> 📌 **关键结果：** 在这个特例中，不需要估计完整的未来 $P_t$ 分布来决定报价。只要价格是鞅，最优策略不依赖绝对价格 $p$，也不依赖具体选择哪一种鞅价格模型；必须校准的是 spread 转移和各报价档的成交强度。

## 📋 第六步：从价值函数生成最优策略表

求出 $\phi_i$ 后，bid 侧最优动作是

$$
\begin{aligned}
(q_t^{b,*},\ell_t^{b,*})
\in\arg\max_{(q^b,\ell^b)\in\mathcal Q_i^b\times[0,\bar\ell]}
\lambda_i^b(q^b)
\Bigg[&\phi_i(t,y+\ell^b)-\phi_i(t,y)\\
&+\left(\frac{i\delta}{2}-\delta\mathbf1_{\{q^b=Bb_+\}}\right)\ell^b
\Bigg].
\end{aligned}
$$

ask 侧最优动作是

$$
\begin{aligned}
(q_t^{a,*},\ell_t^{a,*})
\in\arg\max_{(q^a,\ell^a)\in\mathcal Q_i^a\times[0,\bar\ell]}
\lambda_i^a(q^a)
\Bigg[&\phi_i(t,y-\ell^a)-\phi_i(t,y)\\
&+\left(\frac{i\delta}{2}-\delta\mathbf1_{\{q^a=Ba_-\}}\right)\ell^a
\Bigg].
\end{aligned}
$$

市价单最优数量是

$$
e_t^*
\in
\arg\max_{e\in[-\bar e,\bar e]}
\left[
\phi_i(t,y+e)-\frac{i\delta}{2}|e|-\varepsilon
\right].
$$

只有在 $\phi_i=\mathcal M_\phi\phi_i$ 的干预区域才执行 $e_t^*$；否则使用 bid、ask 两侧的最优限价单动作。最终离线产物可以写成

$$
\Pi^*(t,y,i)
=
\bigl(q^{b,*},\ell^{b,*},q^{a,*},\ell^{a,*},\text{是否干预},e^*\bigr).
$$

这张表才是“从今往后的最优报价规则”。它是在给定模型参数和剩余期限后，对所有可能状态预先求出的反馈控制策略，不是对某一条已实现未来路径的预测。

## ⚡ 第七步：在线观察状态并查表执行

实盘中不需要每次盘口更新都重新解 QVI。在线阶段只需：

1. 观察当前时间 $t$、库存 $Y_t=y$ 和 spread $S_t=i\delta$
2. 查询离线策略表 $\Pi^*(t,y,i)$
3. 更新 bid、ask 报价和挂单量，或发送市价单
4. 成交或 spread 跳变后进入新状态，再次查表

```mermaid
flowchart LR
    accTitle: GP 离线在线执行循环
    accDescr: 离线阶段校准参数并反向求解策略表，在线阶段反复观测时间、库存和 spread，查表后执行最优反馈动作。

    history([📊 历史数据]) --> parameters[🔍 估计外生参数]
    parameters --> solve[⚙️ 反向求解 QVI]
    solve --> table[(📋 策略表)]
    observe([📥 观察 t y i]) --> lookup[🔍 查询动作]
    table --> lookup
    lookup --> execute[⚡ 更新订单]
    execute --> event[📊 成交或状态跳变]
    event --> observe

    classDef input_style fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#3b0764
    classDef process_style fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef result_style fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class history,observe input_style
    class parameters,solve,lookup,execute,event process_style
    class table result_style
```

因此，准确的主线是：

$$
\boxed{
\begin{aligned}
&\text{LOB 历史数据}\\
&\Downarrow\\
&\text{估计 }r_{ij}(t),\lambda_i^b(q),\lambda_i^a(q)\text{ 等外生参数}\\
&\Downarrow\\
&\text{给定效用、库存惩罚、费用与期限}\\
&\Downarrow\\
&\text{离线求解 }\phi_i(t,y)\text{ 和 }\Pi^*(t,y,i)\\
&\Downarrow\\
&\text{在线按当前 }(t,Y_t,S_t)\text{ 查表报价}.
\end{aligned}
}
$$

## ⚠️ 原论文假设与项目边界

| 原论文假设 | 数学含义 | 实现时的限制 |
| --- | --- | --- |
| 小型做市商 | 本方订单不改变 $P,S$ | 不适合直接描述大单市场冲击 |
| $P$ 与 $S$ 独立 | 联合生成元可分开 | 真实市场中价格与 spread 可能相关 |
| 完全成交 | 一次对手单到达即成交全部 $\ell$ | 没有部分成交和 queue-ahead 状态 |
| 独立 Cox 成交过程 | bid、ask 到达由条件强度描述 | 忽略更复杂的订单流依赖 |
| 无延迟监控 | 可即时观察成交并更新控制 | 实盘需额外建模行情与报单延迟 |
| 外生中间价 | 做市商行为不影响 $P$ | 加入 DeepLOB 信号后需扩展状态与生成元 |
| 有限 spread 状态 | $S_t=i\delta$ | 状态数与转移矩阵需重新校准 |

当前项目中的 `GP-style` 若只实现“离散 tick + 库存偏移 + 硬上限 + 日末平仓”，应称为 GP 启发式近似。完整 GP 实现还需要估计 spread 转移、分报价档成交强度，并数值求解上述 QVI。

## 📚 参考资料

[^1]: Guilbaud, F. and Pham, H. (2011). “Optimal High Frequency Trading with Limit and Market Orders.” *arXiv*. https://arxiv.org/abs/1106.5040 。本地 PDF：[Guilbaud_Pham_2011_Optimal_High_Frequency_Trading_with_Limit_and_Market_Orders.pdf](papers/Guilbaud_Pham_2011_Optimal_High_Frequency_Trading_with_Limit_and_Market_Orders.pdf)
