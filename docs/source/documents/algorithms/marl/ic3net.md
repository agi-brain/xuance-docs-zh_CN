# 个体控制的连续通信模型 (IC3Net)

**论文链接:** [**https://arxiv.org/pdf/1812.09755**](https://arxiv.org/pdf/1812.09755).

学习何时通信并有效进行通信在多智能体任务中至关重要。
在之前对 [**CommNet**](./commnet.md) 算法的介绍中，
我们提到它使用连续通信，但仅限于完全协作任务。

IC3Net 算法是 [**CommNet**](./commnet.md) 的改进版本，由纽约大学开发，
是多智能体强化学习领域的知名算法。
它发表在 ICLR 2019。

下表列出了 IC3Net 算法的一些一般特性：

| IC3Net 的特性                                       | 值   | 描述                                                                                                 |
|-----------------------------------------------------|------|-----------------------------------------------------------------------------------------------------|
| 完全去中心化                                          | ❌    | 智能体之间没有通信。                                                                                       |
| 完全中心化                                            | ❌    | 智能体将所有信息发送给中央控制器，控制器将为所有智能体做出决策。                                                        |
| 中心化训练与去中心化执行（CTDE）                            | ✅    | 在训练中使用中央控制器，在执行中放弃。                                                                          |
| 同策略                                                 | ✅    | 评估策略与目标策略相同。                                                                                     |
| 异策略                                                 | ❌    | 评估策略与目标策略不同。                                                                                     |
| 无模型                                                 | ✅    | 不需要准备环境动力学模型。                                                                                   |
| 基于模型                                               | ❌    | 需要环境模型来训练策略。                                                                                     |
| 离散动作                                               | ✅    | 处理离散动作空间。                                                                                         |
| 连续动作                                               | ✅    | 处理连续动作空间。                                                                                         |

## 研究背景与动机

多智能体强化学习（MARL）在现实场景中的应用越来越广泛，
从自动驾驶车队调度到像《星际争霸》这样的复杂游戏中的智能战斗。
智能体之间的有效通信对于实现协作目标至关重要。
然而，早期的多智能体通信算法有三个核心局限性，严重限制了它们对现实场景的适应性：

### 场景适应性不足

经典的 CommNet 算法虽然实现了连续通信的端到端训练，
但仅适用于**完全协作场景**——智能体必须共享所有隐藏状态并追求统一的全局奖励。
然而，大多数现实场景是**混合协作（半协作）**或**竞争性**的：
例如，在篮球比赛中，队友需要合作但也竞争个人得分；
在这种场景下，[CommNet](./commnet.md) 的"无差别完全通信"会导致策略泄露或性能下降。

### 信用分配困境

传统算法采用**全局平均奖励**（所有智能体 receive 相同的奖励），无法区分个体贡献。
例如，在交通路口任务中，如果一辆汽车因错误决策导致碰撞，所有汽车都会受到惩罚。
智能体很难识别自己的错误，导致训练收敛缓慢和可扩展性差
——随着智能体数量的增加，这个问题变得更加突出。

### 缺乏对通信时机的控制

在早期算法中，智能体无论场景需求如何都持续通信：竞争场景中的"无效通信"可能会泄露策略
（例如，猎物向捕食者暴露其位置），而协作场景中的"冗余通信"会增加计算开销。
在现实中，人类会根据场景"在正确的时间通信"（例如，在团队战斗中仅在关键节点传递信息），
但早期算法缺乏这种自适应能力。

## IC3Net 的核心机制

### 基本模型结构：从独立控制到受控通信

IC3Net 的基本框架遵循时间上的"编码-通信-解码"结构，
核心是在 CommNet 的基础上增加"通信门控"和"个体奖励优化"。
首先，定义关键符号：
- $J$：当前环境中的活跃智能体数量（支持动态变化）；
- $o_j^t$：智能体 $j$ 在时间步 $t$ 的局部观测（例如，车辆位置、猎物的视野）；
- $h_j^t, s_j^t$：智能体 $j$ 在时间步 $t$ 的 LSTM 隐藏状态和细胞状态（存储历史信息）；
- $g_j^t$：智能体 $j$ 在时间步 $t$ 的通信门控动作（二元变量：1 = 通信，0 = 不通信）；
- $c_j^t$：智能体 $j$ 在时间步 $t$ 接收到的通信向量；
- $r_j^t$：智能体 $j$ 在时间步 $t$ 的个体奖励（区别于全局奖励）。

模型的核心过程如下：

#### 步骤 1：观测编码和状态更新

首先，让我们描述一个独立控制器模型，其中每个智能体由独立的 [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) 控制。
智能体首先通过编码器 $e(\cdot)$ 将局部观测 $o_j^t$ 转换为特征向量，然后将其输入到 LSTM 中以更新隐藏状态。

对于第 $j$ 个智能体：

$$
h_j^{t+1}, s_j^{t+1} = \text{LSTM}\left(e(o_j^t), h_j^t, s_j^t\right)
$$

其中隐藏状态 $h_t$ 和细胞状态 $s_t$ 在 LSTM 中引入。

编码器 $e(\cdot)$ 是一个全连接神经网络，所有智能体共享 LSTM 参数以确保模型对智能体顺序的置换不变性。

IC3Net 通过允许智能体在离散动作的门控下通信其内部状态来扩展这个独立控制器模型。

上面的公式修改为这种形式：

$$
h_j^{t+1}, s_j^{t+1} = \text{LSTM}\left(e(o_j^t) + c_j^t, h_j^t, s_j^t\right)
$$

其中 $c_j^t$ 将在下文中介绍。

#### 步骤 2：通信门控决策

智能体通过门控网络 $f^g(\cdot)$（线性层 + Softmax）输出通信门控动作 $g_j^t$，以决定是否向其他智能体传输信息：

$$
g_j^t \sim \text{Softmax}\left(f^g(h_j^t)\right)
$$

$g_j^t$ 是一个二元动作（通过采样获得），本质上充当智能体的"通信开关"
——策略网络根据场景效益自主学习何时打开/关闭它。

#### 步骤 3：受控通信向量生成

与 CommNet 的"全局平均广播"不同，IC3Net 的通信向量仅包括"选择通信的智能体"的平均隐藏状态，
然后通过线性变换矩阵 $C$ 将其映射到统一维度：

$$
c_j^{t+1} = C \cdot \frac{1}{\sum_{j' \neq j} g_{j'}^t} \sum_{j' \neq j} g_{j'}^t \cdot h_{j'}^{t+1}
$$

如果没有其他智能体通信（$\sum_{j' \neq j} g_{j'}^t = 0$），则 $c_j^{t+1} = 0$（无通信信号）。
这种设计不仅避免了冗余信息传输，还适应了智能体数量的动态变化。

#### 步骤 4：动作解码和个体奖励优化

智能体的动作通过策略网络 $\pi(\cdot)$ 从隐藏状态 $h_j^t$ 生成：

$$
a_j^t = \pi(h_j^t)
$$

训练目标是最大化每个智能体的累积个体奖励 $\sum_{t=1}^T r_j^t$，而不是全局奖励
——这是信用分配问题的核心解决方案。

## 算法设计：训练框架与基线比较

### 训练算法：REINFORCE + 共享参数优化

IC3Net 使用 [REINFORCE](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf) 算法训练策略网络（包括动作策略 $\pi$ 和门控策略 $f^g$），核心优化目标：

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=1}^T \nabla_\theta \log \pi(a_j^t | h_j^t; \theta) \cdot \sum_{k=t}^T r_j^k\right]
$$

其中 $\theta$ 表示共享参数（在 LSTM、策略网络和门控网络之间共享），以确保模型对智能体顺序的置换不变性。

为了减少训练方差，IC3Net 引入了基线函数 $b(o_j^t)$（状态价值函数），将优化目标修改为：

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=1}^T \nabla_\theta \log \pi(a_j^t | h_j^t; \theta) \cdot \left(\sum_{k=t}^T r_j^k - b(o_j^t)\right)\right]
$$

基线函数由独立的价值网络拟合，以进一步提高训练稳定性。

### IC3Net 的详细参数优化过程

> 特性：独立子网络 + 共享参数

#### 1. 参数初始化

*观测预处理*：使用编码器 $e(\cdot)$ 独立地将每个智能体 $j$ 的局部观测 $o_j^t$ 转换为特征向量，
不拼接全局观测；

*网络参数初始化*：初始化共享参数（由所有子网络重用），包括：
  - LSTM 参数：隐藏状态维度为 128，初始化输入门、遗忘门和输出门的权重（所有智能体共享 LSTM 结构和参数）；
  - 门控网络参数 $f^g(\cdot)$；
  - 通信向量变换权重 $C$；
  - 策略网络 $\pi(\cdot)$ 和基线网络 $b(o_j^t)$；

#### 2. 训练循环阶段

*前向传播*：个体观测 → 受控通信 → 动作输出
（详细信息已在基本模型结构部分展示）。

*个体奖励和梯度计算*

- **步骤 1：个体奖励计算**：根据场景类型分配独立的奖励 $r_j^t$，从而实现精确的信用分配。

- **步骤 2：梯度计算（REINFORCE + 基线）**：基于个体奖励计算"动作策略"和"门控策略"的梯度。

  ① 动作策略梯度：$- \nabla_\theta \log \pi(a_j^t | h_j^t) \cdot (r_j^t - b(o_j^t))$；  
  ② 门控策略梯度：$- \nabla_\theta \log P(g_j^t | h_j^t) \cdot (r_j^t - b(o_j^t))$，其中 $P(g_j^t | h_j^t)$ 是门控动作的概率；
  - 基线网络梯度：$\nabla_\theta (r_j^t - b(o_j^t))^2$，最小化基线与实际奖励之间的差距以减少方差。

*参数更新*

- **步骤 1：梯度聚合**：
累积 $J$ 个智能体的"动作梯度 + 门控梯度 + 基线梯度" $\sum_{j=1}^J \nabla_\theta^{\text{total}}$，以避免单个智能体梯度波动的影响。

- **步骤 2：共享参数更新**：更新所有共享参数。

## 在 XuanCe 中运行 IC3Net

在 XuanCe 中运行 IC3Net 之前，您需要准备 conda 环境并按照 
[**安装步骤**](./../../usage/installation.rst#install-xuance) 安装 ``xuance``。

### 运行内置演示

完成安装后，您可以打开 Python 控制台并使用以下命令直接运行 IC3Net：

```python3
import xuance
runner = xuance.get_runner(method='ic3net',
                           env='mpe',  # 选择：mpe, sc2
                           env_id='simple_spread_v3',  # 选择：simple_spread_v3 等
                           is_test=False)
runner.run()
```

### 使用自定义配置运行

如果您想使用不同的配置运行 IC3Net，可以创建一个新的 ``.yaml`` 文件，例如 ``my_config.yaml``。
然后，通过以下代码块运行 IC3Net：

```python3
import xuance as xp
runner = xp.get_runner(method='ic3net',
                       env='mpe',  # 选择：mpe, sc2
                       env_id='simple_spread_v3',  # 选择：simple_spread_v3 等
                       config_path="my_config.yaml",  # my_config.yaml 文件的路径应该是正确的。
                       is_test=False)
runner.run()  # 或 runner.benchmark()
```

要了解更多关于配置的信息，请访问
[**配置教程**](./../../api/configs/configuration_examples.rst)。

### 使用自定义环境运行

如果您想在 XuanCe 中未包含的自己的环境中运行 XuanCe 的 IC3Net，
您需要按照
[**新环境教程**](./../../usage/custom_env/custom_marl_env.rst) 中的步骤定义新环境。
然后，[**准备配置文件**](./../../usage/custom_env/custom_marl_env.rst#step-2-create-the-config-file-and-read-the-configurations)
``ic3net_myenv.yaml``。

之后，您可以使用以下代码在自己的环境中运行 IC3Net：

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import IC3Net_Agents

configs_dict = get_configs(file_dir="ic3net_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # 创建并行环境。
Agent = IC3Net_Agents(config=configs, envs=envs)  # 从 XuanCe 创建 IC3Net 智能体。
Agent.train(configs.running_steps // configs.parallels)  # 训练模型多个步骤。
Agent.save_model("final_train_model.pth")  # 将模型保存到 model_dir。
Agent.finish()  # 完成训练。
```

## 引用

```{code-block} bash
@article{singh2018learning,
  title={Learning when to communicate at scale in multiagent cooperative and competitive tasks},
  author={Singh, Amanpreet and Jain, Tushar and Sukhbaatar, Sainbayar},
  journal={arXiv preprint arXiv:1812.09755},
  year={2018}
}
```

