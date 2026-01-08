# 深度协调图 (DCG)

**论文链接:** [**https://proceedings.mlr.press/v119/boehmer20a/boehmer20a.pdf**](https://proceedings.mlr.press/v119/boehmer20a/boehmer20a.pdf).

协作多智能体强化学习（MARL）面临**维度灾难**：联合动作空间随着智能体数量的增加呈指数级增长。例如，8 个智能体，每个有 6 个离散动作，会产生超过一百万个联合动作。为了应对这一问题，许多 MARL 方法采用**价值分解**，将中心化 Q 函数分解为更简单的组件（例如，每个智能体一个）。最简单的分解是*完全去中心化*：每个智能体都有一个仅依赖于自身动作的效用函数（如 VDN）。虽然独立效用之和可以表示一些最优策略，但当协调动作的价值显著不同于非协调动作时，这种分解通常会失败。特别地，一个众所周知的问题是**相对过度泛化**：在学习过程中，看到其他智能体随机行动的智能体可能会低估协调动作的价值，导致错过联合最优策略。简而言之，简单的价值分解（如 VDN 和 QMIX）可能缺乏区分高度协调的联合动作的表征能力，从而阻止学习真正的最优解。

```{eval-rst}
.. figure:: ./../../../_static/figures/algo_framework/DCG_value_factorization.png
    :width: 100%    
    :align: center
    
    Figure 1: 价值分解示例（VDN vs DCG vs QTRAN）
```

*图 1 说明了 3 个智能体的不同分解架构：（a）VDN（独立效用），（b）DCG（成对收益），（c）QTRAN（无分解）。此图来自 DCG 论文（图 1）。*

下表列出了 DCG 算法的一些一般特性：

| DCG 的特性                                       | 值   | 描述                                                                 |
|--------------------------------------------------|------|---------------------------------------------------------------------|
| 完全去中心化                                      | ❌    | 智能体通过协调图中的消息传递进行通信，并非完全去中心化且无交互。                        |
| 完全中心化                                        | ❌    | 没有中央控制器；基于局部交互和消息传递进行去中心化决策。                            |
| 中心化训练与去中心化执行（CTDE）                        | ✅    | 在训练中使用中心化信息（例如，智能体的历史、图拓扑），并通过消息传递进行去中心化执行。              |
| 同策略                                             | ❌    | 采用带经验回放的异策略学习（类似于 DQN）。                                      |
| 异策略                                             | ✅    | 评估策略与目标策略不同，利用经验回放提高样本效率。                                 |
| 无模型                                             | ✅    | 直接从交互中学习，不依赖环境动力学模型。                                        |
| 基于模型                                           | ❌    | 不需要环境模型进行策略训练。                                                  |
| 离散动作                                           | ✅    | 设计用于处理离散动作空间，使用低秩近似来提高可扩展性。                              |
| 连续动作                                           | ❌    | 主要针对离散动作空间；连续动作支持不是核心特性。                                   |

## 1. 协作 MARL 中的挑战

协作 MARL 的核心挑战是**随联合动作大小的扩展**。随着智能体数量 *n* 的增长，联合动作空间 |A₁×…×Aₙ| 变得过大。没有结构的情况下，在此空间上估计中心化 Q 函数是不可行的。简单的去中心化，即每个智能体仅最大化局部效用函数，提供了**可处理的优化**，但经常错过必要的协调：这种分解可以表示至少一个最优确定性策略，但由于相对过度泛化等博弈论问题，在实践中**可能无法学习到它**。即使像 QMIX 这样的高级分解，允许中心 Q 是每个智能体效用的**单调混合**，也无法表示所有可能的价值景观（单调性约束可能排除表示一些最优联合价值）。因此，**有限的表征能力**和**协调困难**是 MARL 中的关键问题。DCG 的开发旨在通过引入更灵活的分解决来解决这些问题，同时保持可处理的学习和执行。

## 2. VDN 和 QMIX 的局限性

**价值分解网络（VDN）**和**QMIX**是流行的 MARL 方法，它们对联合 Q 函数施加特定的分解。VDN 简单地求和各个智能体效用：

$$
Q_{VDN}(s,\mathbf{a}) = \sum_{i=1}^n f_i(a_i \mid \tau^i) 
$$

其中每个智能体 *i* 具有依赖于其局部观测历史 $\tau^i$ 的效用 $f_i$。在 VDN 中，每个智能体可以通过独立最大化 $f_i$ 来选择其动作，这在计算上是高效的。然而，这种表示**过于受限**：它假设最好的联合动作是通过每个智能体采取其单独最好的动作获得的。在许多任务中，最好的联合结果需要为了协调而牺牲即时个体效用。如果函数类 $f_i$ 无法捕获来自协调的额外奖励，学习算法**无法区分**高价值的协调动作和非协调动作，并且不会收敛到真正的最优策略。

QMIX 通过引入**混合网络** $ϕ$ 来推广 VDN，该网络将所有智能体效用作为输入并产生联合 Q 值：

$$
Q_{QMIX}(s,\mathbf{a}) = ϕ\Big(s,\; f_1(a_1 \mid \tau^1),\ldots,f_n(a_n \mid \tau^n)\Big)\,.
$$

关键的是，$ϕ$ 在每个效用输入中被约束为**单调的**。这种单调性确保最大化各个 $f_i$ 仍然产生最好的联合动作。QMIX 可以表示比 VDN 更大类别的价值函数，特别是通过使用全局状态 $s$ 作为 $ϕ$ 的输入，但它**无法捕获非单调交互**。在实践中，具有硬协调的任务通常涉及在个体效用中非单调的价值函数，因此 QMIX（和 VDN）无法学习最优策略。总之，尽管 VDN/QMIX 是可处理的，它们的受限分解在真正的协作任务中导致失败模式（如相对过度泛化）。

无边的 DCG（VDN）最终必然会失败（p < −1）。

```{eval-rst}
.. figure:: ./../../../_static/figures/algo_framework/DCG-Performance-Comparison.png
    :width: 100%    
    :align: center
    
    Figure 2: 过度泛化任务中的性能比较
```

*图 2 显示了不同模型在相对过度泛化任务中的表现，其中 8 个智能体猎捕 8 个猎物。全连接的 DCG 能够表示联合动作的价值，从而在更大的 p 值下获得更好的性能，而无边的 DCG（VDN）在较低的 p 值下失败。没有参数共享的 CG 由于样本效率低下而学习非常缓慢。*

## 3. 深度协调图（DCG）：图分解和消息传递

DCG 通过使用**协调图（CG）**来定义联合 Q 函数的分解，从而克服了这些局限性。协调图是无向图 $\mathcal{G}=\langle\mathcal{V},\mathcal{E}\rangle$，其中每个顶点对应于一个智能体，每条边 $\{i,j\}\in \mathcal{E}$ 表示智能体 $i$ 和 $j$ 之间的成对*收益函数*。在 DCG 中，Q 函数被分解为沿图边的**每个智能体效用**和**成对收益**：

$$
Q^{\mathrm{CG}}(s_{t},\boldsymbol{a}):=\frac{1}{|\mathcal{V}|}\sum_{v^{i}\in\mathcal{V}}f^{i}(a^{i}|s_{t})+\frac{1}{|\mathcal{E}|}\sum_{\{i,j\}\in\mathcal{E}}f^{ij}(a^{i},a^{j}|s_{t}).
$$

这里每个效用 $f_i$ 依赖于智能体 $i$ 的动作 $a_i$（可能还有其观测历史），每个收益 $f_{ij}$ 依赖于智能体 $i$ 和 $j$ 的联合动作（可选地依赖于状态）。归一化因子 $1/|\mathcal{V}|$ 和 $1/|\mathcal{E}|$ 是可选的缩放；核心思想是联合价值是个体和成对贡献的平均值。当 $\mathcal{E}=\emptyset$ 时，这简化为 VDN。添加每条边都会增加容量：**每个成对收益可以建模独立效用无法建模的协调效应**。实际上，DCG 学习包含成对智能体交互的更丰富的价值函数类。

一旦联合 Q 函数被 CG 分解，**最优联合动作**（或贪婪动作）通常不能由每个智能体单独找到。相反，DCG 使用协调图上的**最大和消息传递**（一种信念传播形式）来（近似）计算 $Q_{DCG}$ 的联合 argmax。在树结构图中，最大和更新保证收敛到真正最大值。在循环图中，DCG 使用启发式*消息归一化*（减去均值项）来改善收敛性。DCG 的关键优势是**表示能力足够强大以捕获协调**，但推理仍然是局部的。与完全中心化 Q 相比，DCG 的消息传递每次决策需要 $O(km(n+m)|E|)$ 时间（对于 $k$ 次迭代，$n$ 个智能体，每个 $m$ 个动作），这对于稀疏图是可处理的。重要的是，DCG 采用**参数共享**：所有收益网络 $f_{ij}$ 共享相同的权重（以智能体 ID 为条件），所有效用 $f_i$ 共享权重。这种共享，以及收益的可选低秩分解，显著提高了样本效率。

## 4. DCG 的核心 Q 函数和设计原则
### 1. DCG 的核心 Q 函数
DCG 基于协调图实现价值分解，其核心联合 Q 函数如下：

$$
q_{\theta\phi\psi}^{\mathrm{DCG}}(\boldsymbol{\tau}_{t},\boldsymbol{a}) := \frac{1}{|\mathcal{V}|}\sum_{i=1}^{n}\overbrace{f_{\theta}^{v}(a^{i}|\boldsymbol{h}_{t}^{i})}^{f_{i,a^{i}}^{\mathrm{V}}} + \frac{1}{2|\mathcal{E}|}\sum_{\{i,j\}\in\mathcal{E}}\underbrace{f_{\phi}^{e}(a^{i},a^{j}|\boldsymbol{h}_{t}^{i},\boldsymbol{h}_{t}^{j})+f_{\phi}^{e}(a^{j},a^{i}|\boldsymbol{h}_{t}^{j},\boldsymbol{h}_{t}^{i})}_{f_{\{i,j\},a^{i},a^{j}}^{\mathrm{E}}}
$$

其中：
- $q^{DCG}(a|\tau_t)$ 表示在智能体历史 $\tau_t$ 下执行联合动作 $a$ 的期望折扣奖励和；
- $|\mathcal{V}|$ 是智能体总数（协调图中的顶点数），$|\mathcal{E}|$ 是智能体对总数（协调图中的边数）；
- $f_\theta^v(a^i|h_t^i)$ 是智能体 $i$ 的个体效用函数（由 $\theta$ 参数化，输入为 RNN 隐藏状态 $h_t^i$）；
- $f_\phi^e(a^i,a^j|h_t^i,h_t^j)$ 是智能体 $i$ 和智能体 $j$ 之间的成对收益函数（由 $\phi$ 参数化，输入为两个智能体的 RNN 隐藏状态 $h_t^i$ 和 $h_t^j$）。


### 2. DCG 的核心设计原则
1. 收益函数仅依赖于局部信息（智能体 $i$ 和 $j$ 的历史 $\tau_t^i$ 和 $\tau_t^j$）；
2. 所有收益/效用函数共享参数（通过共同的 RNN）；
3. 对收益矩阵应用低秩近似；
4. 支持跨图泛化（置换不变性）。

## 5. DCG 算法和伪代码

下面我们概述 DCG 中的三个关键计算过程。

### 5.1. 标注协调图（效用和收益计算）

```{eval-rst}
.. figure:: ./../../../_static/figures/pseucodes/pseucode-DCG-1.png
    :width: 100%   
    :align: center
    
    pseudocode 1: 标注协调图伪代码
```

此伪代码计算每个智能体和智能体对的效用和收益张量。它基于每个智能体的先前状态、观测和动作更新其隐藏状态，然后基于协调图计算个体效用和成对收益。

---

### 5.2. Q 值计算

```{eval-rst}
.. figure:: ./../../../_static/figures/pseucodes/pseucode-DCG-2.png
    :width: 100%  
    :align: center
    
    pseudocode 2: Q 值计算伪代码
```

此伪代码通过组合个体智能体效用和成对收益来计算联合 Q 值。计算的 Q 值用于评估联合动作的质量，这在 DCG 框架中选择最优动作至关重要。

---

### 5.3. 使用消息传递的贪婪动作选择

```{eval-rst}
.. figure:: ./../../../_static/figures/pseucodes/pseucode-DCG-3.png
    :width: 100%    
    :align: center
    
    pseudocode 3: 使用消息传递的贪婪动作选择伪代码
```

此伪代码实现了在协调图上使用消息传递的贪婪动作选择。消息被迭代更新，并由每个智能体使用来选择最大化其对联合 Q 值贡献的动作，确保 DCG 方法中的协调决策。

## 在 XuanCe 中运行 DCG

在 XuanCe 中运行 DCG 之前，您需要准备 conda 环境并按照 
[**安装步骤**](./../../usage/installation.rst#install-xuance) 安装 ``xuance``。

```python3
import xuance
# 为 DCG 算法创建运行器
runner = xuance.get_runner(method='dcg',
                           env='sc2',  # 选择：sc2, mpe
                           env_id='3m',  # 选择：3m, 2m_vs_1z, 8m, 1c3s5z, 2s3z, 25m, 5m_vs_6m, 8m_vs_9m, MMM2 等
                           is_test=False)  # False 用于训练，True 用于测试
runner.run()  # 开始运行（或 runner.benchmark() 用于基准测试）
```

### 使用自定义配置运行

如果您想使用不同的配置运行 DCG，可以创建一个新的 ``.yaml`` 文件，例如 ``my_config.yaml``。
然后，通过以下代码块运行 DCG：

```python3
import xuance as xp
# 为 DCG 算法创建运行器
runner = xp.get_runner(method='DCG',
                       env='sc2',  # 选择：sc2, mpe
                       env_id='3m',  # 选择：3m, 2m_vs_1z, 8m, 1c3s5z, 2s3z, 25m, 5m_vs_6m, 8m_vs_9m, MMM2 等
                       config_path="my_config.yaml",  # my_config.yaml 文件的路径应该是正确的。
                       is_test=False)  # False 用于训练，True 用于测试
runner.run()  # 开始运行（或 runner.benchmark() 用于基准测试）
```

要了解更多关于配置的信息，请访问 
[**配置教程**](./../../api/configs/configuration_examples.rst)。

### 使用自定义环境运行

如果您想在 XuanCe 中未包含的自己的环境中运行 XuanCe 的 DCG， 
您需要按照
[**新环境教程**](./../../usage/custom_env/custom_drl_env.rst) 中的步骤定义新环境。
然后，[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``dcg_myenv.yaml``。

之后，您可以使用以下代码在自己的环境中运行 DCG：

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_MULTI_AGENT_ENV 
from xuance.environment import make_envs
from xuance.torch.agents.multi_agent_rl.dcg_agents import DCG_Agents 

configs_dict = get_configs(file_dir="DCG_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_MULTI_AGENT_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # 创建并行环境。
Agent = DCG_Agents(config=configs, envs=envs)  # 从 XuanCe 创建 DCG 智能体。
Agent.train(configs.running_steps // configs.parallels)  # 训练模型多个步骤。
Agent.save_model("final_train_model.pth")  # 将模型保存到 model_dir。
Agent.finish()  # 完成训练。
```

## 引用

```{code-block} bash
@InProceedings{pmlr-v119-boehmer20a,
  title = {Deep Coordination Graphs},
  author = {Boehmer, Wendelin and Kurin, Vitaly and Whiteson, Shimon},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  pages = {980--991},
  year = {2020},
  editor = {III, Hal Daumé and Singh, Aarti},
  volume = {119},
  series = {Proceedings of Machine Learning Research},
  month = {13--18 Jul},
  publisher = {PMLR},
  pdf = {http://proceedings.mlr.press/v119/boehmer20a/boehmer20a.pdf},
  url = {https://proceedings.mlr.press/v119/boehmer20a.html},
}
```

