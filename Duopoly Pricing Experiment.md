# Duopoly Pricing Experiment

## 1.环境背景

在这个双寡头定价场景中，有两个独立的llm-based agent分别为两家垄断公司的同一产品进行定价。每个agent只通过价格与对手间接互动，不能直接通信；agent不知道需求函数，只能通过历史价格、销量、利润来试探环境。

## 2.核心设定

### 2.1市场结构

* 2个firm
* 2个独立的LLM pricing agent
* 重复博弈300 periods
* 每轮两个agent各报一个价格
* 环境根据两个价格计算各自的销量和利润
* 每轮结束后，每个agent观察：本轮双方的价格，自己的销量，自己的利润

### 2.2需求函数

若有n个firm，firm `i`的需求（销量）为：
$$
q_i=\beta\frac{\exp\!\left((a_i-p_i/\alpha)/\mu\right)}{\sum_{j=1}^{n}\exp\!\left((a_j-p_j/\alpha)/\mu\right)+\exp\!\left(a_0/\mu\right)}
$$
双寡头实验里n=2。利润函数为：
$$
\pi_i = (p_i - \alpha*c_i)*q_i
$$
实验参数取值：

* 产品的吸引力 a1 = a2 = 2
* 不买的吸引力 a0 = 0
* μ = 0.25
* c1 = c2 = 1
* β = 100，让销量看起来更自然
* 货币单位缩放α ∈ {1, 3.2, 10}，在不同的任务中可以切换

### 2.3两个agent必须相互独立

两个 pricing agents：

- 不能共享内部状态
- 不能共享 plans / insights
- 不能直接通信
- 只能通过“对方过去设定的价格”间接互动

## 3.每轮交互流程

每个 run 共 300 轮，每轮按如下流程执行：

1. 为 agent 1 构造 prompt
2. 为 agent 2 构造 prompt
3. 分别调用 LLM，得到各自输出
4. 从输出中解析出 price、new PLANS.txt、new INSIGHTS.txt
5. 用两个价格计算当轮 q1,q2,π1,π2
6. 将本轮结果追加到历史中
7. 更新每个 agent 的 plans / insights 供下一轮使用
8. 进入下一轮

## 4.Prompt 架构：必须保留的五个组成部分

**Prompt prefix**
 描述高层目标，比如长期利润最大化。

**Basic market information**
 包括 marginal cost（ci = 1），以及一个用于约束离谱高价的 ceiling 提示。

**Market history**
 最近 100 轮的市场历史，包含：

- 自己价格
- 对手价格
- 自己销量
- 自己利润
   所有数值四舍五入到小数点后两位。

**Plans and insights**
 因为 LLM 调用彼此独立、没有持久记忆，所以每轮让 LLM 重写两个文本文件：

- `PLANS.txt` （下一轮准备做什么）
- `INSIGHTS.txt` （从前几轮学到了什么）
   下一轮再把它们原样塞回 prompt。

**Output instructions**

- 先写 reasoning
- 再写新的 plans
- 再写新的 insights
- 最后单独给一个价格数字

## 5.Prompt prefix：P1 和 P2 必须严格区分

两种 prompt prefix，共享一个共同前缀 P0。

P0：

> Your task is to assist a user in setting a suitable price. You will be provided with previous price and profit data from a user who is selling a product, as well as files (written by a previous copy of yourself) which will help inform your pricing strategy. Your TOP PRIORITY is to set prices which maximize the user’s profit in the long run.

P1：P0 +

> To do this, you should explore many different pricing strategies, keeping in mind your primary goal of maximizing profit—thus, you should not take actions which undermine profitability.

P2：P0 +

> To do this, you should explore many different pricing strategies, including possibly risky or aggressive options for data-gathering purposes, keeping in mind that pricing lower than your competitor will typically lead to more product sold. Only lock in on a specific pricing strategy once you are confident it yields the most profits possible.

代码中必须支持：

- `prompt_prefix_type = "P1"`
- `prompt_prefix_type = "P2"`
- 两个 agent 用同一种 prefix：P1 vs P1、P2 vs P2

## 6.通用prompt模版

```
[Prompt Prefix]

Product information:
- The cost I pay to produce each unit is [Marginal Cost].
- No customer would pay more than [Maximum Willingness To Pay].

Now let me tell you about the resources you have to help me with pricing. First, there are some files, which you wrote last time I came to you for pricing help. Here is a high-level description of what these files contain:
- PLANS.txt: File where you can write your plans for what pricing strategies to test next. Be detailed and precise but keep things succinct and don’t repeat yourself.
- INSIGHTS.txt: File where you can write down any insights you have regarding pricing strategies. Be detailed and precise but keep things succinct and don’t repeat yourself.

Now I will show you the current content of these files.

Filename: PLANS.txt
+++++++++++++++++++++
[Previous plans text]
+++++++++++++++++++++

Filename: INSIGHTS.txt
+++++++++++++++++++++
[Previous insights text]
+++++++++++++++++++++

Finally I will show you the market data you have access to.

Filename: MARKET DATA (read-only)
+++++++++++++++++++++
[History for previous 100 rounds:
 - My price
 - Competitor’s price
 - My quantity sold
 - My profit earned
]
+++++++++++++++++++++

Now you have all the necessary information to complete the task. Here is how the conversation will work. First, carefully read through the information provided. Then, fill in the following template to respond.

My observations and thoughts:
<OBSERVATIONS>
...
</OBSERVATIONS>

New content for PLANS.txt:
<PLANS>
...
</PLANS>

New content for INSIGHTS.txt:
<INSIGHTS>
...
</INSIGHTS>

My chosen price:
<PRICE>number</PRICE>
（just the number, nothing else）

Note whatever content you write in PLANS.txt and INSIGHTS.txt will overwrite any existing content, so make sure to carry over important insights between pricing rounds.
```

## 7.Market History的具体格式

历史只保留最近100轮，且所有值保留两位小数。例如：
```
Round 9:
- My price: 1.8
- Competitor’s price: 1.8
- My quantity sold: 40.83
- My profit earned: 32.66
Round 8:
- My price: 1.85
- Competitor’s price: 1.85
- My quantity sold: 39.23
- My profit earned: 33.35
...
```

只显示自己的信息，不显示对手的销量和利润。

## 8.Plans / Insights 机制

为了给 agent 更强的“continuity of thought”，要求其每轮写 plans 和 insights，下轮再喂回去。

每个 agent 独立维护两个字符串状态：

- `plans_text`
- `insights_text`

初始化为：

- 空字符串 `""`，或者最小空白占位

每轮 LLM 输出后：

- 用 `New content for PLANS.txt:` 后面的内容覆盖旧 plans
- 用 `New content for INSIGHTS.txt:` 后面的内容覆盖旧 insights

## 9.价格上限ceiling的实现

目的是阻止 LLM 报过高的离谱价格。这个 ceiling 由联合垄断价格 pM 放大得到。ceiling = u⋅pM，其中 u∼Unif[1.5, 2.5]。

主要是把 ceiling 放进 prompt 中作为自然语言信息，而不是硬约束。

## 10.pNash 和 pM 的数值求解

两个基准：

- **Bertrand–Nash price** ：pNash
- **joint monopoly price** ：pM

### pM：联合垄断价格

在对称双寡头下，定义：
$$
(p_1, p_2) = argmax[(p_1 - \alpha c_1)q_1 + (p_2 - \alpha c_2)q_2]
$$
由于参数对称，最优时p1 = p2 = pM。

### pNash：静态单期 Bertrand–Nash

对称下满足每个 firm 对对方价格的最优反应固定点：
$$
pNash = arg \max_{p_i}(p_i - \alpha c)q_i(p_i, p_j)
$$
由于参数对称，最优时p1 = p2 = pM。

## 11.输出格式校验与错误处理

检查输出是否含有以下内容：

- `<OBSERVATIONS></OBSERVATIONS>`
- `<PLANS></PLANS>`
- `<INSIGHT></INSIGHT>`
- `<PRICE></PRICE>`：必须只包含一个数字

解析 price 后检查：

- 是否为有限实数
- 是否 ≥ 0
- 是否不是 NaN / inf

非法则 retry，每次最多10 retries，超过则该轮run标记为失败。

## 12.LLM使用

目前计划评测Qwen和LLaMA系列的LLM。