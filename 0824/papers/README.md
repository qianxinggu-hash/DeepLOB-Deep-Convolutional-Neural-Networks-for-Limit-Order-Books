# AS 与 GP 做市论文

_0824 方向增强做市实验使用的本地参考论文。_

---

## 📚 论文索引

| 简称 | 论文 | 本地文件 | 用途 |
| --- | --- | --- | --- |
| AS | Avellaneda and Stoikov (2008), _High-frequency trading in a limit order book_[^1] | [PDF](Avellaneda_Stoikov_2008_High_Frequency_Trading_in_a_Limit_Order_Book.pdf) | 连续价格、库存风险与最优双边报价基准 |
| GP | Guilbaud and Pham (2013), _Optimal high frequency trading with limit and market orders_[^2] | [PDF](Guilbaud_Pham_2013_Optimal_High_Frequency_Trading_with_Limit_and_Market_Orders.pdf) | 离散 tick、随机价差、限价单与市价单控制框架 |

## ⚠️ 使用边界

项目中的 GP 策略是面向现有逐笔数据的离散近似，并非原论文 QVI 系统的完整数值解。具体实现边界与实验结论见 [`project_methodology.md`](../project_methodology.md) 和 [`reconstruction_and_results.md`](../reconstruction_and_results.md)。

---

[^1]: Avellaneda, M., and Stoikov, S. (2008). "High-frequency trading in a limit order book." _Quantitative Finance_. https://math.nyu.edu/inmemoriam/avellaneda/HighFrequencyTrading.pdf

[^2]: Guilbaud, F., and Pham, H. (2013). "Optimal high frequency trading with limit and market orders." _Quantitative Finance_. https://arxiv.org/abs/1106.5040
