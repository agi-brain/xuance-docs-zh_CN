# 独立近端策略优化 (IPPO)

**论文链接:** [**https://proceedings.neurips.cc/paper_files/paper/2022**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html).

独立近端策略优化（IPPO）是一种用于协作多智能体强化学习（MARL）的分布式策略梯度方法。尽管近端策略优化（PPO）由于其稳定性和效率已成为单智能体强化学习的主流算法之一，但它在多智能体环境中的应用长期以来一直被低估。传统观点认为，与异策略方法（如 MADDPG、QMIX 等）相比，PPO 类算法的样本效率较低，难以处理高维和非平稳的多智能体训练任务。

然而，最近的研究表明，通过合理配置实现细节，基于 PPO 的多智能体算法可以在各种基准任务中实现甚至超越主流异策略方法的性能（Yu 等人，NeurIPS 2022）。其中，作为典型架构，IPPO 体现了"独立学习的去中心化执行"思想。其最初的设计意图是仅通过局部观测和共享奖励信号实现多个智能体的有效协调，而不依赖中心化评判器或显式通信机制。

这种方法特别适用于具有部分可观测性、异构性或有限通信的实际场景。它具有良好的可扩展性和工程部署潜力，因此已成为现代 MARL 基线系统的重要组成部分。

下表列出了 IPPO 算法的一些一般特性：

| IPPO 的特性                                       | 值   | 描述                                                                                                 |
|---------------------------------------------------|------|-----------------------------------------------------------------------------------------------------|
| 完全去中心化                                        | ✅    | 智能体之间没有通信。                                                                                       |
| 完全中心化                                          | ❌    | 智能体将所有信息发送给中央控制器，控制器将为所有智能体做出决策。                                                        |
| 中心化训练与去中心化执行（CTDE）                            | ❌    | 在训练中使用中央控制器，在执行中放弃。                                                                          |
| 同策略                                               | ✅    | 评估策略与目标策略相同。                                                                                     |
| 异策略                                               | ❌    | 评估策略与目标策略不同。                                                                                     |
| 无模型                                               | ✅    | 不需要准备环境动力学模型。                                                                                   |
| 基于模型                                             | ❌    | 需要环境模型来训练策略。                                                                                     |
| 离散动作                                             | ✅    | 处理离散动作空间。                                                                                         |
| 连续动作                                             | ❌    | 处理连续动作空间。                                                                                         |

## 背景与研究动机

### 传统去中心化方法的性能瓶颈

早期的去中心化算法（如 IAC 和独立 Q 学习（IQL））采用"每个智能体独立学习"模式。虽然它们避免了对全局信息的依赖，但有两个主要缺陷：首先，**环境非平稳性加剧**——每个智能体的策略更新改变了其他智能体的"环境"，导致价值估计偏差的累积；其次，**信用分配模糊**——全局奖励无法直接与单个智能体的动作关联，容易出现"搭便车"或"过度探索"等问题。

### PPO 在多智能体场景中的"应用差距"

单智能体 PPO 通过**重要性采样裁剪**（限制策略更新的幅度）和**广义优势估计（GAE）**（平衡优势函数的偏差和方差）等设计实现了稳定性和样本利用之间的平衡，并支持并行采样。然而，MARL 领域长期以来存在两种认知偏见：

**样本效率误解**：认为同策略方法需要实时收集新样本，其效率远低于异策略方法（如 QMIX 和 MADDPG）；

**对非平稳性的恐惧**：认为多智能体环境中的动态变化会放大 PPO 的梯度噪声，导致训练崩溃。
这些认知使 PPO 的多智能体扩展（如 IPPO）长期处于研究边缘，少数尝试（如早期 IPPO 变体）也没有系统验证其在复杂任务中的有效性。

## 核心思想

每个智能体独立运行标准的近端策略优化（PPO）算法。策略网络参数在它们之间不共享（除非智能体是同质的），它们也不依赖中心化价值函数。

### 去中心化决策机制

每个智能体 $i \in \{1, \ldots, n\}$ 维护一个独立的策略网络 $\pi_{\theta_i}(a_i|o_i)$，仅基于其局部观测 $o_i$ 做出决策动作 $a_i$，满足实际系统中每个实体自主运行的需求。

### 在 IID（独立同分布）假设下的策略更新

尽管整体环境是非平稳的（即其他智能体策略的变化导致环境的动态变化），IPPO 仍然假设在每个训练批次内，每个智能体的经验可以被视为独立同分布样本，并据此进行策略更新。

### 由共享奖励驱动的协作激励

所有智能体共享相同的全局奖励函数 $R_t$，从而形成共同的优化目标。这种设置确保了个体策略改进的方向与集体性能的改进一致，避免了竞争博弈带来的协调问题。

尽管 IPPO 不使用中心化价值函数来减少方差，但它仍然继承了 PPO 的关键优势——裁剪代理目标，有效控制策略更新步长并防止因策略突变导致的训练崩溃。

## 局限性

### 多智能体规模敏感性

在完全去中心化架构中，智能体只能基于局部观测做出决策，缺乏推断队友策略意图的能力。当智能体数量超过 3 时，"协作碎片化"问题加剧——个体智能体的动作意图无法对齐，导致联合策略的协调性急剧下降。

### 信息不对称导致的次优解

在需要全局协作的部分可观测场景中，IPPO 由于缺乏全局状态信息而容易产生次优策略。局部观测无法充分反映环境的真实状态（例如，SMAC 中涉及"空地单位协调战斗"的场景）。智能体难以判断整体情况，导致单位协作不匹配和资源分配不合理等问题。

### 样本效率仍低于中心化异策略方法

尽管 IPPO 在同策略算法中表现良好，但与主流异策略方法（如 QMIX、VDAC）相比，其样本效率仍有显著差距。同策略范式需要实时收集新样本，无法重用历史经验；相比之下，异策略方法可以通过回放缓冲区长期存储和利用样本，降低环境交互成本。

### 环境非平稳性的持续干扰

每个智能体的策略更新改变了其他智能体的"训练环境"，导致价值函数估计偏差的累积。尽管 IPPO 通过"减少训练轮数"和"价值归一化"等措施缓解了这一问题，但在超大规模智能体场景（例如，100+ 智能体的集群）中，训练仍然容易出现振荡。

## 在 XuanCe 中运行 IPPO

在 XuanCe 中运行 IPPO 之前，您需要准备 conda 环境并按照 
[**安装步骤**](./../../usage/installation.rst#install-xuance) 安装 ``xuance``。

### 运行内置演示

完成安装后，您可以打开 Python 控制台并使用以下命令直接运行 IPPO：

```python3
import xuance
runner = xuance.get_runner(method='ippo',
                    env='mpe',  
                    env_id='simple_spread_v3',  
                    is_test=False)
runner.run() 
```
### 使用自定义配置运行

如果您想使用不同的配置运行 IPPO，可以创建一个新的 ``.yaml`` 文件，例如 ``my_config.yaml``。
然后，通过以下代码块运行 IPPO：

```python3
import xuance
runner = xuance.get_runner(method='ippo',
                       env='mpe', 
                       env_id='simple_spread_v3',  
                       config_path="my_config.yaml",  # my_config.yaml 文件的路径应该是正确的。
                       is_test=False)
runner.run()
```
### 使用自定义环境运行

如果您想在 XuanCe 中未包含的自己的环境中运行 XuanCe 的 IPPO， 
您需要按照
[**新环境教程**](./../../usage/custom_env/custom_drl_env.rst) 中的步骤定义新环境。
然后，[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``ippo_myenv.yaml``。

之后，您可以使用以下代码在自己的环境中运行 IPPO：

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import IPPO_Agents

configs_dict = get_configs(file_dir="ippo_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs) 
Agent = IPPO_Agents(config=configs, envs=envs) 
Agent.train(configs.running_steps // configs.parallels)  
Agent.save_model("final_train_model.pth") 
Agent.finish()  # 完成训练。
```

## 引用

```{code-block} bash
@article{yu2022surprising,
  title={The surprising effectiveness of ppo in cooperative multi-agent games},
  author={Yu, Chao and Velu, Akash and Vinitsky, Eugene and Gao, Jiaxuan and Wang, Yu and Bayen, Alexandre and Wu, Yi},
  journal={Advances in neural information processing systems},
  volume={35},
  pages={24611--24624},
  year={2022}
}
```

