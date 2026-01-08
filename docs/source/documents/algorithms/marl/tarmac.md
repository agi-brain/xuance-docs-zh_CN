# 目标多智能体通信模型 (TarMAC)

**论文链接:** [**https://proceedings.mlr.press/v97/das19a/das19a.pdf**](https://proceedings.mlr.press/v97/das19a/das19a.pdf).

## 1. 架构定位与核心目标

TarMAC（目标多智能体通信）是一种多智能体强化学习（MARL）架构，旨在解决传统多智能体通信中的低效问题。其核心目标是：
- 实现**无监督目标通信**：智能体仅通过任务奖励独立学习"与谁通信"和"通信什么"，无需额外的通信监督。
- 支持**多轮交互**：通过在一个时间步内重复通信来适应复杂任务（例如，多路口交通调度、3D 导航），以传达完整信息。
- 采用**CTDE（中心化训练与去中心化执行）框架**：使用中心化信息进行稳定训练，使用去中心化决策进行实际部署。

## 2. 核心特性

| TarMAC 的特性                                       | 值   | 描述                                                                 |
|-----------------------------------------------------|------|---------------------------------------------------------------------|
| 完全去中心化                                        | ❌    | 依赖目标消息传递进行协作；完全去中心化且无交互会失败。                               |
| 完全中心化                                          | ❌    | 没有中央控制器；智能体通过局部观测和消息进行去中心化决策。                             |
| 中心化训练与去中心化执行（CTDE）                            | ✅    | 使用中心化信息（例如，智能体的隐藏状态）进行训练；使用目标消息进行去中心化执行。               |
| 同策略                                               | ✅    | 同步批处理执行器-评判器（无经验回放）；评估策略与目标策略匹配。                          |
| 异策略                                               | ❌    | 无经验回放；评估策略与目标策略无差异。                                        |
| 无模型                                               | ✅    | 从智能体-环境交互中学习；不需要环境动力学模型。                                  |
| 基于模型                                             | ❌    | 不需要环境模型进行策略训练。                                              |
| 离散动作                                             | ✅    | 设计用于离散动作（例如，SHAPES 中的"上/下"）；支持可扩展性。                        |
| 连续动作                                             | ❌    | 主要用于离散动作；连续动作支持不是核心功能。                                    |

## 3. 整体架构

```{eval-rst}
.. figure:: ./../../../_static/figures/algo_framework/TarMAC_Schematic.png
    :width: 100%    
    :align: center
    
    Figure 1: TarMAC 的多智能体架构示意图（左：智能体策略流程；右：目标通信机制）
```

该架构由两个相互连接的端到端可微分组件组成

### 3.1 智能体策略流程（左侧）

在每个时间步 $t$：
- **输入**：每个智能体接收两种类型的数据：① 局部观测 $\omega_i^t$（例如，SHAPES 中的 5×5 局部网格，House3D 中的 224×224 第一人称图像）；② 来自前一个时间步的聚合消息 $c_i^t$。
- **核心计算**：一层 GRU（门控循环单元）更新隐藏状态 $h_i^t$（编码"观测-动作-消息"的完整历史）。
- **输出**：从 $h_i^t$ 生成两个独立输出：① 离散环境动作 $a_i^t$（例如，交通路口中的"加速/刹车"）；② 目标通信消息 $m_i^t$（包含签名和值组件）。

### 3.2 目标通信机制（右侧）

- **消息发送**：发送者通过 $m_i^t$ 的"签名 $k_i^t$"编码预期接收者的属性。
- **消息接收**：接收者从其隐藏状态 $h_j^{t+1}$ 生成查询向量 $q_j^{t+1}$，通过将 $q_j^{t+1}$ 与发送者的签名匹配来计算注意力权重，并将相关消息聚合为 $c_j^{t+1}$（用于下一个时间步）。

## 4. 核心模块 1：目标通信

这是 TarMAC 的关键创新，通过三个步骤实现无监督目标：

### 4.1 步骤 1：消息构建

每个智能体的消息 $m_i^t$ 结合了"签名"（用于接收者匹配）和"值"（用于实际内容）：

$$
m_i^t = \left[ \underbrace{k_i^t}_{\text{签名（预期接收者属性）}} \parallel \underbrace{v_i^t}_{\text{值（实际通信内容）}} \right].
$$  

- $k_i^t$：16 维向量（在所有实验中统一），编码目标属性（例如，SHAPES 中的"红色目标相关"，交通路口中的"西向东行驶"）。
- $v_i^t$：32 维向量（在所有实验中统一），携带实际信息（例如，SHAPES 中的智能体坐标，House3D 中的目标方向）。

### 4.2 步骤 2：注意力权重计算

接收者 $j$ 从其隐藏状态 $h_j^{t+1}$ 生成 $d_k$ 维查询向量 $q_j^{t+1}$（其中 $d_k=16$，文件中统一设置的签名维度），然后计算与所有发送者 $k_1^t, k_2^t, ..., k_N^t$ 的签名匹配度，并在 softmax 归一化后获得注意力权重 $\alpha_j$。公式为：

$$
\alpha_j = \text{softmax}\left[ \frac{q_j^{t+1^T}k_1^t}{\sqrt{d_k}}, ..., \frac{q_j^{t+1^T}k_N^t}{\sqrt{d_k}} \right].
$$  

- 分子：点积 $q_j^{t+1^T}k_i^t$ 反映接收者的信息需求（编码在 $q_j^{t+1}$ 中）与发送者的消息属性（编码在 $k_i^t$ 中）之间的匹配度；值越大表示相关性越高，例如"搜索红色目标"的查询向量与"红色属性"的签名向量匹配。
- 分母：$\sqrt{d_k}$（签名维度 $d_k$ 的平方根）用于避免高维向量点积引起的数值饱和，确保 softmax 函数可以灵活分配权重，这与文件中的参数设置和设计逻辑一致。

### 4.3 步骤 3：消息聚合

接收者使用注意力权重聚合发送者的值向量，以获得下一个时间步的输入消息 $c_j^{t+1}$：

$$
c_j^{t+1} = \sum_{i=1}^N \alpha_{ji} v_i^t.
$$  

- 验证：论文显示此步骤过滤了无关消息——例如，针对蓝色的智能体将权重 >0.8 分配给"蓝色属性"消息，将 <0.05 分配给无关消息。

## 5. 核心模块 2：多轮通信

单轮通信无法处理复杂任务，因此 TarMAC 添加了多轮交互（在一个时间步内）：

### 5.1 核心公式

隐藏状态被更新以累积多轮信息：

$$
{h^{\prime}}_j^t=\tanh\left(W_{h\to h^{\prime}}[
\begin{array}
{c}c_j^{t+1}\parallel h_j^t
\end{array}]\right).
$$

- 输入："当前聚合消息 $c_j^{t+1}$"和"初始隐藏状态 $h_j^t$"的拼接（保留历史和新信息）。
- 变换：$W_{h \to h'}$（可学习线性矩阵）映射到 128 维 GRU 隐藏状态；$\tanh$ 将值约束到 $[-1, 1]$ 以避免梯度爆炸。

### 5.2 迭代规则

- 最优轮数：实验确认 2 轮效果最佳——第 1 轮传达一般信息（例如，"目标在北方"），第 2 轮细化细节（例如，"绕过门到北方"）。超过 2 轮不会提供性能提升，但会增加训练时间。

## 6. 关键实验验证

实验参数统一：RMSProp 优化器（学习率 $7 \times 10^{-4}$），折扣因子 $\gamma = 0.99$，5 次独立运行的平均值。

### 6.1 实验 1：SHAPES

```{eval-rst}
.. figure:: ./../../../_static/figures/algo_framework/TarMAC_Timing.png
    :width: 100%    
    :align: center
    
    Figure 2: SHAPES 中智能体通信的时间图（t=1 到 t=21）
```
- 任务：4 个智能体在 50×50 网格中搜索多色目标。
- 核心结果：TarMAC 的成功率（85.8±2.5%）优于"无通信"（69.1±4.6%）和"无注意力的通信"（82.4±2.1%）。
- 视觉洞察：在 $t=2$，针对红色的智能体专注于那些观测到红色的智能体；在 $t=21$，所有智能体在到达目标后使用自注意力。

### 6.2 实验 2：交通路口

```{eval-rst}
.. figure:: ./../../../_static/figures/algo_framework/TarMAC_Traffic.png
    :width: 100%    
    :align: center
    
    Figure 3: 交通路口模型解释（a：刹车概率；b：注意力位置；c：动态团队适应
```

- 任务：汽车在 4 个双向路口避免碰撞。
- 核心结果：2 轮通信（97.1±1.6%）优于 CommNet（78.9±3.4%）和 1 轮通信（84.6±3.2%）。
- 视觉洞察：汽车在路口附近刹车，将注意力集中在高风险区域（"第一个路口之后"），并适应动态团队规模（关注的汽车数量与总汽车数量相关）。

## 在 XuanCe 中运行 TarMAC

在 XuanCe 中运行 TarMAC 之前，您需要准备 conda 环境并按照 
[**安装步骤**](./../../usage/installation.rst#install-xuance) 安装 ``xuance``。

```python3
import xuance
# 为 TarMAC 算法创建运行器
runner = xuance.get_runner(method='tarmac',
                           env='sc2',  # 选择：sc2, mpe
                           env_id='3m',  # 选择：3m, 2m_vs_1z, 8m, 1c3s5z, 2s3z, 25m, 5m_vs_6m, 8m_vs_9m, MMM2 等。
                           is_test=False)  # False 用于训练，True 用于测试
runner.run()  # 开始运行（或 runner.benchmark() 用于基准测试）
```

### 使用自定义配置运行

如果您想使用不同的配置运行 TarMAC，可以创建一个新的 ``.yaml`` 文件，例如 ``my_config.yaml``。
然后，通过以下代码块运行 TarMAC：

```python3
import xuance as xp
# 为 TarMAC 算法创建运行器
runner = xp.get_runner(method='TarMAC',
                       env='sc2',  # 选择：sc2, mpe
                       env_id='3m',  # 选择：3m, 2m_vs_1z, 8m, 1c3s5z, 2s3z, 25m, 5m_vs_6m, 8m_vs_9m, MMM2 等。
                       config_path="my_config.yaml",  # my_config.yaml 文件的路径应该是正确的。
                       is_test=False)  # False 用于训练，True 用于测试
runner.run()  # 开始运行（或 runner.benchmark() 用于基准测试）
```

要了解更多关于配置的信息，请访问 
[**配置教程**](./../../api/configs/configuration_examples.rst)。

### 使用自定义环境运行

如果您想在 XuanCe 中未包含的自己的环境中运行 XuanCe 的 TarMAC， 
您需要按照
[**新环境教程**](./../../usage/custom_env/custom_drl_env.rst) 中的步骤定义新环境。
然后，[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``tarmac_myenv.yaml``。

之后，您可以使用以下代码在自己的环境中运行 TarMAC：

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_MULTI_AGENT_ENV 
from xuance.environment import make_envs
from xuance.torch.agents.multi_agent_rl.tarmac_agents import TarMAC_Agents 

configs_dict = get_configs(file_dir="TarMAC_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_MULTI_AGENT_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # 创建并行环境。
Agent = TarMAC_Agents(config=configs, envs=envs)  # 从 XuanCe 创建 TarMAC 智能体。
Agent.train(configs.running_steps // configs.parallels)  # 训练模型多个步骤。
Agent.save_model("final_train_model.pth")  # 将模型保存到 model_dir。
Agent.finish()  # 完成训练。
```

## 引用

```{code-block} bash
@InProceedings{das2019tarmac,
  title     = {TarMAC: Targeted Multi-Agent Communication},
  author    = {Das, Abhishek and Gervet, Th{\'e}ophile and Romoff, Joshua and Batra, Dhruv and Parikh, Devi and Rabbat, Mike and Pineau, Joelle},
  booktitle = {International Conference on Machine Learning},
  pages     = {1538--1546},
  year      = {2019},
  publisher = {PMLR},
  pdf       = {http://proceedings.mlr.press/v97/das19a/das19a.pdf},
  url       = {https://proceedings.mlr.press/v97/das19a.html}
}
```

