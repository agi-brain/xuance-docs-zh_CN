# 加权 Q-混合网络 (WQMIX)

**论文链接:** [**https://proceedings.neurips.cc/paper_files/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf**](https://proceedings.neurips.cc/paper_files/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf).

今天，我们继续介绍加权 QMIX（WQMIX）算法——一种基于价值的多智能体强化学习（MARL）算法。
顾名思义，WQMIX 是 [QMIX](./qmix.md) 的改进版本。
如果您不熟悉 [QMIX](./qmix.md) 算法，建议先参考分析 [QMIX](./qmix.md) 算法的文档。

WQMIX 算法也是由牛津大学 Whiteson 研究实验室的成员开发的，
发表在 NeurIPS 2020。

下表列出了 WQMIX 算法的一些一般特性：

| WQMIX 的特性                                       | 值   | 描述                                                                                                 |
|----------------------------------------------------|------|-----------------------------------------------------------------------------------------------------|
| 完全去中心化                                        | ❌    | 智能体之间没有通信。                                                                                       |
| 完全中心化                                          | ❌    | 智能体将所有信息发送给中央控制器，控制器将为所有智能体做出决策。                                                        |
| 中心化训练与去中心化执行（CTDE）                            | ✅    | 在训练中使用中央控制器，在执行中放弃。                                                                          |
| 同策略                                               | ❌    | 评估策略与目标策略相同。                                                                                     |
| 异策略                                               | ✅    | 评估策略与目标策略不同。                                                                                     |
| 无模型                                               | ✅    | 不需要准备环境动力学模型。                                                                                   |
| 基于模型                                             | ❌    | 需要环境模型来训练策略。                                                                                     |
| 离散动作                                             | ✅    | 处理离散动作空间。                                                                                         |
| 连续动作                                             | ❌    | 处理连续动作空间。                                                                                         |

## 研究背景与动机

在用于协作任务的 MARL 环境中，多个智能体共享单个奖励信号。
这通常会导致"懒惰智能体"和"信用分配"问题。
为了解决这些问题，[VDN](./vdn.md) 算法提出使用简单求和来分解系统的总价值函数：
$Q_{tot}(\boldsymbol{\tau},\boldsymbol{u})=\sum_{i=1}^nQ_i(\tau_i,u_i)$。
这种价值函数分解方法简单直接，
但其对"单调性"约束的过度遵守严重限制了其近似非线性函数的能力。
为了克服这一局限性，提出了 QMIX 算法。
它使用**混合网络**来拟合 $Q_{tot}^*$，显著增强了网络的非线性函数逼近能力
，同时满足去中心化策略的"单调性"要求。

然而，正如我们之前对 [QMIX](./qmix.md) 的分析所指出的，[QMIX](./qmix.md) 满足的单调性约束是**充分但不必要**的条件。
这意味着存在一些场景，其中 [QMIX](./qmix.md) 无法准确拟合价值函数。
例如，当智能体的动作相互影响时（即，一个智能体的决策必须考虑其他智能体的动作），
[QMIX](./qmix.md) 无法考虑这种相互依赖性。因此，**QMIX 的函数表示能力仍然有限**。

因此，这项研究的主要目标是进一步摆脱对 [QMIX](./qmix.md) 函数表示能力的约束。

## QMIX 算子：定义与性质

### QMIX 算子的定义：$\mathcal{T}_{\mathrm{Qmix}}^*$

为了简化理解和分析，作者假设所有智能体观测全局状态 $s$ 并使用 Q 表进行分析。

首先，他们定义 $Q_{tot}$ 所属的函数空间：

$$
\mathcal{Q}^{mix}=\{Q_{tot}|Q_{tot}(s,\boldsymbol{u})=f_{s}(Q_{1}(s,u_{1}),\cdots,Q_{n}(s,u_{n})),\frac{\partial f_{s}}{\partial Q_{a}}\geq0,Q_{a}(s,u)\in\mathbb{R}\}
$$

正如在 [QMIX](./qmix.md) 的文档中分析的，QMIX 算法的设计确保
网络可以以任意精度近似 $\mathcal{Q}^{mix}$ 中的任何函数。
QMIX 可以表述为以下优化问题：

$$
\arg\min_{q\in\mathcal{Q}^{mix}}\sum_{\boldsymbol{u}\in\boldsymbol{U}}(\mathcal{T}^*Q_{tot}(s,\boldsymbol{u})-q(s,\boldsymbol{u}))^2,\forall s\in S.(1)
$$

这里，$\mathcal{T}^*$ 表示贝尔曼最优算子：

$$
\mathcal{T}^*Q(s,\boldsymbol{u}):=\mathbb{E}[r+\gamma\max_{\boldsymbol{u}^{\prime}}Q(s^{\prime},\boldsymbol{u}^{\prime})].(2)
$$

作者将公式$`(1)`$中的优化问题定义为**QMIX 算子**，记为 $\mathcal{T}_{\mathrm{Qmix}}^*$。

它可以被视为两个算子的组合：

$$
\mathcal{T}_{\mathrm{Qmix}}^*=\Pi_{\mathrm{Qmix}}\mathcal{T}^*.(3)
$$

其中 $\Pi_{\mathrm{Qmix}}$ 定义为：

$$
\Pi_{\mathrm{Qmix}}Q:=\arg\min_{q\in\mathcal{Q}^{mix}}\sum_{\boldsymbol{u}\in \boldsymbol{U}}(Q(s,\boldsymbol{u})-q(s,\boldsymbol{u}))^2.(4)
$$

从几何角度来看，$\Pi_{\mathrm{Qmix}}Q$ 表示函数空间 $\mathcal{Q}^{mix}$ 中
距离函数 $Q$ 最近的点（通过 L2 范数测量）。

### $\mathcal{T}_{\mathrm{Qmix}}^*$ 算子的性质

由于公式$`(1)`$中的优化是在 $`\mathcal{Q}^{mix}`$ 空间内执行的，
[QMIX](./qmix.md) 在某些情况下可能无法找到 $`\mathcal{T}^*`$ 的不动点。
相反，它只能找到 $`\mathcal{Q}^{mix}`$ 中最接近不动点的次优解。
因此，公式$`(1)`$的优化结果可能不是唯一的，
$`\mathcal{T}_{\mathrm{Qmix}}^*`$ 将随机返回一个 $`q`$ 函数作为最终解。

> 性质 1：$\mathcal{T}_{\mathrm{Qmix}}^*$ 不是收缩映射。

熟悉泛函分析的人会认识到收缩映射原理（也称为 Banach 不动点定理）。
该定理保证了非空完备度量空间上的自映射不动点的存在性和唯一性。
实际上，算子 $\mathcal{T}^*$ 是收缩映射，其不动点的存在性和唯一性得到保证
——这构成了 [**Q 学习**](https://link.springer.com/article/10.1007/bf00992698) 算法的理论基础。

那么，为什么 $\mathcal{T}_{\mathrm{Qmix}}^*$ 不是收缩映射呢？

为了解释这一点，作者提出了一个简单的案例：

> 假设表 1 中的左矩阵是 Q 函数的奖励矩阵 $`Q^*`$，
> 它无法由 $`\mathcal{Q}^{mix}`$ 中的任何价值函数表示。
> 使用 $`\Pi_{\mathrm{Qmix}}Q`$，我们可能获得表 1 中间或右侧所示的 $`Q_{tot}`$ 矩阵
> ——两者都允许智能体获得最大奖励 $`r = 1`$。
> 因此，$`\mathcal{T}_{\mathrm{Qmix}}^*`$ 算子计算的 $`Q_{tot}`$ 可能不是唯一的。
> 因此它缺乏收缩映射的"唯一不动点"性质，本身不是收缩映射。

```{eval-rst}
.. figure:: ./../../../_static/figures/algo_framework/wqmix_table1.png
    :width: 80%
    :align: center
  
    Table 1. 非单调奖励矩阵（左）和 QMIX 算子返回的两个可能解（中和右）
```

这种现象是因为算子在 $\mathcal{Q}^{mix}$ 空间内求解问题，
其关键特征是"单调性"。当不动点表现出"非单调性"时，
它位于 $\mathcal{Q}^{mix}$ 之外，迫使算法找到近似解（如表 1 中的两个解）。

由于收缩映射定义在完备度量空间上，QMIX 算子 $`\mathcal{T}_{\mathrm{Qmix}}^*`$
——定义在 $`\mathcal{Q}^{mix}`$ 空间上——不是收缩映射。

这为 [QMIX](./qmix.md) 的局限性提供了更深入的解释。

> 性质 2：QMIX 中由 $Q_{tot}$ 最大化的联合动作并不总是正确的。

作者指出，可能存在 Q 函数，使得 $\arg\max\Pi_{\mathrm{Qmix}}Q\neq\arg\max Q$。
如果您理解性质 1，这个性质将是直观的。

例如，考虑一个 2 智能体、3 动作任务。表 2 中的左矩阵是真实的奖励矩阵 $`Q^*`$，
右矩阵是 $`\Pi_{\mathrm{Qmix}}`$ 算子返回的 $`Q_{tot}`$。
根据 $`Q^*`$，智能体对可以通过选择相应的动作组合获得最大奖励 $r = 8$。
然而，为了满足"单调性约束"，$\Pi_{\mathrm{Qmix}}Q^*$ 可能将 $r = 8$ 位置的奖励拟合为 $-12$，
使智能体能够达到的最大奖励为 $r = 0$。

```{eval-rst}
.. figure:: ./../../../_static/figures/algo_framework/wqmix_table2.png
    :width: 60%
    :align: center
  
    Table 2. 真实奖励矩阵（左）和 QMIX 算子返回的奖励矩阵（右）。
```

这种现象也源于 $\mathcal{Q}^{mix}$ 函数空间的局限性。

> 性质 3：QMIX 可能低估某些联合动作的价值。

此性质与性质 2 密切相关。
不正确的 argmax 计算不可避免地导致价值函数估计中的进一步错误。
例如，表 2 中 $r = 8$ 的奖励被错误地估计为 $-12$。

这三个性质是 [QMIX](./qmix.md) 算法固有的缺陷，与计算性能、
探索机制或网络参数设置无关。
即使在理想条件下，[QMIX](./qmix.md) 也表现出如此显著的缺陷。
在实际训练中——存在更多不可控因素——情况可能更糟。

## 加权 QMIX 算子

既然我们已经确定了 [QMIX](./qmix.md) 的缺陷，我们可以直接解决它们。作者的推理如下：

> 公式$`(1)`$中的优化问题遍历状态空间中的所有状态和联合动作空间中的所有动作。
> 作者认为，如果真实奖励是"非单调的"，
> 对所有动作值使用相等权重（即均匀加权）求和是不合理的。
>
> 以表 2 中的右矩阵为例。由于 $Q_{tot}$ 必须满足单调性约束，
> 算法只能通过两种方式调整动作值来将 $-12$ 的错误估计修正为 $8$：
> 要么增加对应于 $r = -12$（次优动作）的动作值，要么减少对应于 $r = 0$（次优动作）的动作值。
> 两种选择都会增加整体误差，因此 $Q_{tot}$ 避免此类调整并接受次优解。
>
> 基于这一见解，作者提出对各个动作值进行加权，以减轻此类调整对整体误差的影响。

通过向 $\Pi_{\mathrm{Qmix}}$ 算子添加权重函数 $w$，导出了新算子：

$$
\Pi_wQ:=\arg\min_{q\in\mathcal{Q}^{mix}}\sum_{\boldsymbol{u}\in U}w(s,\boldsymbol{u})(Q(s,\boldsymbol{u})-q(s,\boldsymbol{u}))^2.(5)
$$

这里，权重函数 $w$ : $S\times U\to(0,1]$ 在 [QMIX](./qmix.md) 的损失函数中加权每个联合动作的重要性。
$w$ 的输入空间不限于状态-联合动作对；也可以包含其他因素。
值得注意的是，当 $w(s,\boldsymbol{u})\equiv1$ 时，$\Pi_{w}\Leftrightarrow\Pi_{\mathrm{Qmix}}$。

### 权重函数 $w(s,\boldsymbol{u})$ 的选择

#### 第一个权重函数：理想化中心加权

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\dpi{200}\large%20w(s,%5Cboldsymbol%7Bu%7D)=%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bll%7D1,%26%20%5Cboldsymbol%7Bu%7D=%5Cboldsymbol%7Bu%7D%5E*=%5Carg%5Cmax_%7B%5Cboldsymbol%7Bu%7D%7DQ(s,%5Cboldsymbol%7Bu%7D)%5C%5C%5Calpha,%26%20%5Ctext%7Botherwise.%7D%5Cend%7Barray%7D%5Cright.%5Ctag%7B6%7D" alt="w(s,u) definition">
</p>

这种加权方法简单直接，但需要遍历整个联合动作空间来计算 $\arg\max$，使其在实际使用中不切实际。
正如后面讨论的，作者在实际训练中使用近似方法来解决这个问题。

论文为此权重函数提供了以下结论：

> 定理 1. 设 $w$ 是公式$`(6)`$中定义的理想化中心加权。
> $\exists\alpha>0$ 使得对于任何 $Q$，$\arg\max\Pi_wQ=\arg\max Q$。

该定理保证了"理想化中心加权"的存在性，
确保 $\arg\max\Pi_wQ$ 不会产生错误的动作（与 [QMIX](./qmix.md) 不同）。

#### 第二个权重函数：乐观加权

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\dpi{200}\large%20w(s,%5Cboldsymbol%7Bu%7D)=%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bll%7D1,%26%20Q_%7Btot%7D(s,%5Cboldsymbol%7Bu%7D)%3CQ(s,%5Cboldsymbol%7Bu%7D)%5C%5C%5Calpha,%26%20%5Ctext%7Botherwise.%7D%5Cend%7Barray%7D%5Cright.%5Ctag%7B7%7D" alt="w(s,u) piecewise">
</p>

该权重函数对所有被低估的动作值分配更高的权重（$1$），
对所有被高估的动作值分配较低的权重（$\alpha$）。
这确保尽可能准确地估计动作值，因此称为"乐观加权"。

类似地，作者为此权重函数提供了以下结论：

> 定理 2. 设 $w$ 是公式$`(7)`$中定义的理想化中心加权。
> $\exists\alpha>0$ 使得对于任何 $Q$，$\arg\max\Pi_wQ=\arg\max Q$。

定理 2 也保证了"乐观加权"的存在性，
确保对 $Q_{tot}$ 的 argmax 操作不会产生错误的动作。

由于篇幅限制，定理 1 和定理 2 的证明此处不详述；
请参阅 [**原始论文**](https://proceedings.neurips.cc/paper_files/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Supplemental.pdf) 中的补充材料。

### 加权 QMIX 算子

根据定理 1 和定理 2 的结论，
上述两个权重函数确保任何 Q 函数（包括 $Q^*$）的联合动作输出的准确性。

两种权重函数的设计都需要 $Q^*$ 进行计算。
因此，必须学习一个额外的 $`\hat{Q}^{*}`$（来近似 $`Q^*`$）。
然而，在拟合期间计算 $`\arg\max\hat{Q}^{*}`$ 需要搜索整个联合动作空间
——这是计算上不可行的任务。为了解决这个问题，作者利用 $`Q_{tot}`$ 的"单调性"
并提议使用 $`\arg\max Q_{tot}`$ 生成动作，然后由 $`\hat{Q}^{*}`$ 估计。

$\hat{Q}^{*}$ 函数使用以下算子更新：

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\dpi{200}\large%20%5Cmathcal%7BT%7D_w%5E*%5Chat%7BQ%7D%5E*(s,%5Cboldsymbol%7Bu%7D):=%5Cmathbb%7BE%7D%5Br%2B%5Cgamma%5Chat%7BQ%7D%5E*(s%5E%5Cprime,%5Carg%5Cmax_%7B%5Cboldsymbol%7Bu%7D%5E%5Cprime%7DQ_%7Btot%7D(s%5E%5Cprime,%5Cboldsymbol%7Bu%7D%5E%5Cprime))%5D.%5Ctag%7B8%7D" alt="Tw Q definition">
</p>

比较公式 $`(8)`$ 与前面介绍的公式 $`(2)`$，$`\mathcal{T}_w^*`$ 和 $`\mathcal{T}^*`$ 之间的关键区别是

$`\mathcal{T}_w^*`$ 不通过直接最大化 $`\hat{Q}^{*}`$ 来选择动作；

相反，它最大化单调函数 $`Q_{tot}\in\mathcal{Q}^{mix}`$。注意，当 $`w(s,\boldsymbol{u})\equiv1`$ 时，
$`\Pi_w\nLeftrightarrow\Pi_{\mathrm{Qmix}}`$。

类似地，$Q_{tot}$ 使用以下算子更新：

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\dpi{200}\large%20%5Cmathcal%7BT%7D_%7B%5Cmathrm%7BWQMIX%7D%7D%5E*Q_%7Btot%7D:=%5CPi_w%5Cmathcal%7BT%7D_w%5E*%5Chat%7BQ%7D%5E*.%5Ctag%7B9%7D" alt="T_WQMIX definition">
</p>

这个 $\mathcal{T}_{\mathrm{WQMIX}}^*$ 是加权 QMIX 算子。

为了确保 $\mathcal{T}_{\mathrm{WQMIX}}^*$ 收敛到最优策略，作者提供了以下结论：
> 推论 1. 设 $`w`$ 为理想化中心或乐观加权，
> 则 $`\exists\alpha>0`$ 使得 $`\mathcal{T}_w^*`$ 的唯一不动点是 $`Q_{tot}`$。
> 此外，$`\Pi_{w}Q^{*}\subseteq\mathcal{Q}^{mix}`$ 恢复最优策略，且 $`\max\Pi_wQ^*(s,\cdot)=\max Q^*(s,\cdot)`$。

## 算法设计

以上对 WQMIX 的分析假设了理想条件。
下面，我们讨论如何在部分可观测性下使用深度神经网络实现 WQMIX。

基于前面的分析，WQMIX 算法由三个组件组成：$Q_{tot}$、$\hat{Q}^*$ 和 $w(s,\boldsymbol{u})$。
算法框架如图 1 所示。

```{eval-rst}
.. figure:: ./../../../_static/figures/algo_framework/wqmix_framework.png
    :width: 80%
    :align: center
  
    Figure 1. 深度强化学习设置中加权 QMIX 算法的实现。
```

### $Q_{tot}$ 函数

$Q_{tot}$ 的结构设计与原始 [QMIX](./qmix.md) 算法中的几乎相同。
它通过最小化以下损失进行训练：

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\dpi{200}\large%20%5Csum_%7Bi%3D1%7D%5Eb%20w(s,%5Cboldsymbol%7Bu%7D)(Q_%7Btot%7D(%5Cboldsymbol%7B%5Ctau%7D,%5Cboldsymbol%7Bu%7D,s)-y_i)%5E2.%5Ctag%7B10%7D" alt="loss function eq 10">
</p>

这里，$y_{i}:=r+\gamma\hat{Q}^{*}(s^{\prime},\boldsymbol{\tau}^{\prime},\arg\max_{\boldsymbol{u}^{\prime}}Q_{tot}(\boldsymbol{\tau}^{\prime},\boldsymbol{u}^{\prime},s^{\prime}))$ 

### $\hat{Q}^*$ 函数

作者使用类似于 QMIX 的结构实现 $`\hat{Q}^*`$（参见图 1 的左侧面板）。
然而，[QMIX](./qmix.md) 中的混合网络被替换为标准前馈网络。
该网络以所有智能体的 Q 函数 $`\{Q_i(\tau_i^{\prime},u_i)\}_{i=1}^n`$ 
和全局状态 $`s^{\prime}`$ 作为输入，并输出 $`\hat{Q}^*`$。

$\hat{Q}^*$ 函数通过最小化以下损失函数进行训练：

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\dpi{200}\large%20%5Csum_%7Bi%3D1%7D%5Eb(%5Chat%7BQ%7D%5E*(%5Cboldsymbol%7B%5Ctau%7D,%5Cboldsymbol%7Bu%7D,s)-y_i)%5E2.%5Ctag%7B11%7D" alt="Q loss eq 11">
</p>

$y_i$ 的计算与公式$`(10)`$中的相同。

### 权重函数 $w(s,\boldsymbol{u})$

计算两个权重函数都需要在联合动作空间上最大化 $\hat{Q}^*$，
当智能体数量很大时，这在计算上是昂贵的。
因此，作者使用近似方法来解决这个问题。

#### 中心加权 QMIX (CW-QMIX)

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\dpi{200}\large%20w(s,%5Cboldsymbol%7Bu%7D)=%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bll%7D1%20&%20Q_%7Btot%7D(%5Cboldsymbol%7B%5Ctau%7D,%5Cboldsymbol%7Bu%7D,s)%3C%20y_i%5C%5C%5Calpha%20&%20%5Ctext%7Botherwise.%7D%5Cend%7Barray%7D%5Cright.%5Ctag%7B12%7D" alt="w(s,u) eq 12">
</p>

这里，$`\boldsymbol{\hat{u}}^{*}=\arg\max_{\boldsymbol{u}}Q_{tot}(\boldsymbol{\tau},\boldsymbol{u},s)`$。
如果 $`y_{i}>\hat{Q}^{*}(s,\boldsymbol{\tau}`$，$`u`$ 可以近似地被视为最优联合动作。

与公式$`(6)`$相比，公式$`(12)`$将 $`\mathcal{T}_w^*\hat{Q}^{*}(s,\boldsymbol{\tau},\boldsymbol{\hat{u}^{*}})`$ 替换为 $`\hat{Q}^{*}(s,\boldsymbol{\tau},\boldsymbol{\hat{u}^{*}})`$。
基于公式$`(12)`$的加权 QMIX 算法称为"中心加权 QMIX (CW-QMIX)"。

### 乐观加权 QMIX (OW-QMIX)

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\dpi{200}\large%20w(s,%5Cboldsymbol%7Bu%7D)=%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bll%7D1%20&%20Q_%7Btot%7D(%5Cboldsymbol%7B%5Ctau%7D,%5Cboldsymbol%7Bu%7D,s)%3C%20y_i%5C%5C%5Calpha%20&%20%5Ctext%7Botherwise.%7D%5Cend%7Barray%7D%5Cright.%5Ctag%7B13%7D" alt="w(s,u) eq 13">
</p>

基于公式$`(13)`$的加权 QMIX 算法称为"乐观加权 QMIX (OW-QMIX)"。
与公式$`(7)`$不同，它不需要近似。

以上涵盖了加权 QMIX 算法的理论和设计方面。
WQMIX 不是通过修改 [QMIX](./qmix.md) 的"单调性约束"来增强其函数表示能力，
而是使用权重函数根据每个动作的重要性对联合动作空间中的每个动作进行加权。
这允许单调函数（满足 [QMIX](./qmix.md) 的约束）通过加权 QMIX 算子映射到非单调函数。

## 结论

本文解决了 [QMIX](./qmix.md) 函数表示能力的局限性。
通过对理想条件下的 [QMIX](./qmix.md) 的分析，作者提出了加权 QMIX 算法。
理论上，WQMIX 保证单调函数（由 [QMIX](./qmix.md) 输出）可以通过动作加权映射到非单调价值函数。
这使得 WQMIX 能够学习最优策略，并避免了 [QMIX](./qmix.md) 收敛到次优策略的倾向。

为了学习权重函数，WQMIX 还学习一个不受"单调性"约束的 $`\hat{Q}^*`$ 函数，
确保其收敛到 $`Q^*`$。
然而，在实际训练中，近似方法和 $`\hat{Q}^*`$ 结构的设计可能导致性能下降。

作者指出，WQMIX 仍有进一步改进的空间——特别是在权重函数的设计方面。
本文使用的权重函数很简单，仅考虑全局状态和联合动作信息，
取值要么为 1，要么为 $\alpha$。
因此，更复杂的权重函数代表了 WQMIX 未来研究的有希望方向。

## 在 XuanCe 中运行 WQMIX

在 XuanCe 中运行 WQMIX 之前，您需要准备 conda 环境并按照 
[**安装步骤**](./../../usage/installation.rst#install-xuance) 安装 ``xuance``。

### 运行内置演示

完成安装后，您可以打开 Python 控制台并使用以下命令直接运行 WQMIX：

```python3
import xuance
runner = xuance.get_runner(method='wqmix',
                           env='mpe',  # 选择：football, mpe, sc2
                           env_id='simple_spread_v3',  # 选择：simple_spread_v3 等
                           is_test=False)
runner.run()
```

### 使用自定义配置运行

如果您想使用不同的配置运行 WQMIX，可以创建一个新的 ``.yaml`` 文件，例如 ``my_config.yaml``。
然后，通过以下代码块运行 WQMIX：

```python3
import xuance as xp
runner = xp.get_runner(method='wqmix',
                       env='mpe',  # 选择：football, mpe, sc2
                       env_id='simple_spread_v3',  # 选择：simple_spread_v3 等
                       config_path="my_config.yaml",  # my_config.yaml 文件的路径应该是正确的。
                       is_test=False)
runner.run()  # 或 runner.benchmark()
```

要了解更多关于配置的信息，请访问 
[**配置教程**](./../../api/configs/configuration_examples.rst)。

### 使用自定义环境运行

如果您想在 XuanCe 中未包含的自己的环境中运行 XuanCe 的 WQMIX， 
您需要按照
[**新环境教程**](./../../usage/custom_env/custom_marl_env.rst) 中的步骤定义新环境。
然后，[**准备配置文件**](./../../usage/custom_env/custom_marl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``wqmix_myenv.yaml``。

之后，您可以使用以下代码在自己的环境中运行 WQMIX：

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import WQMIX_Agents

configs_dict = get_configs(file_dir="wqmix_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # 创建并行环境。
Agent = WQMIX_Agents(config=configs, envs=envs)  # 从 XuanCe 创建 WQMIX 智能体。
Agent.train(configs.running_steps // configs.parallels)  # 训练模型多个步骤。
Agent.save_model("final_train_model.pth")  # 将模型保存到 model_dir。
Agent.finish()  # 完成训练。
```

## 引用

```{code-block}
@article{rashid2020weighted,
  title={Weighted qmix: Expanding monotonic value function factorisation for deep multi-agent reinforcement learning},
  author={Rashid, Tabish and Farquhar, Gregory and Peng, Bei and Whiteson, Shimon},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={10199--10210},
  year={2020}
}
```

