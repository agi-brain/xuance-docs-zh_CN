# 多智能体近端策略优化 (MAPPO)

**论文链接:** [**https://proceedings.neurips.cc/paper_files/paper/2022**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html).

## 背景与研究动机

在多智能体强化学习（MARL）领域，协作任务中的协作决策问题一直是近年来的研究热点。传统的多智能体训练方法通常依赖于中心化训练与去中心化执行（CTDE）范式，旨在使用全局信息来提高训练稳定性，同时确保每个智能体在执行阶段仅基于局部观测做出独立决策。

尽管 QMIX 和 MADDPG 等异策略算法在多个基准任务中取得了显著成果，但由于良好的收敛性和稳定性，同策略方法在大规模分布式系统中显示出巨大潜力。其中，近端策略优化（PPO）作为单智能体场景中最成功的策略梯度算法之一，因其简单、高效和超参数鲁棒性强而广受青睐。

然而，在多智能体环境中，PPO 的应用长期以来一直受到质疑，主要基于两个假设：一是其样本效率低于异策略方法；二是单智能体环境中的实现经验难以迁移到多智能体场景。最近的研究表明，经过适当调整后，基于 PPO 的多智能体算法，即多智能体 PPO（MAPPO），在各种协作任务中表现出色，甚至超越了主流异策略基线。这一发现促使学术界重新审视 PPO 在 MARL 中的地位，MAPPO 因此成为一类重要的强基线算法。

下表列出了 MAPPO 算法的一些一般特性：

| MAPPO 的特性                                       | 值   | 描述                                                                                                 |
|----------------------------------------------------|------|-----------------------------------------------------------------------------------------------------|
| 完全去中心化                                        | ❌    | 智能体之间没有通信。                                                                                       |
| 完全中心化                                          | ❌    | 智能体将所有信息发送给中央控制器，控制器将为所有智能体做出决策。                                                        |
| 中心化训练与去中心化执行（CTDE）                            | ✅    | 在训练中使用中央控制器，在执行中放弃。                                                                          |
| 同策略                                               | ✅    | 评估策略与目标策略相同。                                                                                     |
| 异策略                                               | ❌    | 评估策略与目标策略不同。                                                                                     |
| 无模型                                               | ✅    | 不需要准备环境动力学模型。                                                                                   |
| 基于模型                                             | ❌    | 需要环境模型来训练策略。                                                                                     |
| 离散动作                                             | ✅    | 处理离散动作空间。                                                                                         |
| 连续动作                                             | ✅    | 处理连续动作空间。                                                                                         |


## 关键实现细节

MAPPO 本质上是标准近端策略优化（PPO）算法在反事实多智能体策略梯度（CTDE）框架下的自然扩展。

### 策略与价值函数的分离

MAPPO 采用两个独立的神经网络：

**策略网络** $ \pi_\theta(u^a|o^a) $：参数化智能体的动作分布，以局部观测 $ o^a $ 作为输入并输出动作概率。对于同质智能体（例如，SMAC 中的相同单位类型），采用参数共享机制，通过输入"智能体 ID"来区分不同智能体。

**价值网络** $ V_\phi(s) $：仅用于训练阶段的方差减少，以全局状态 $ s $（例如，MPE 中所有智能体观测的拼接，SMAC 中环境提供的全局战场信息）作为输入并输出状态价值 $ V(s) $。

### 参数共享

在同质智能体环境中（如大多数 SMAC 地图），MAPPO 通常采用参数共享策略：所有智能体共享同一组策略和价值网络参数。这不仅显著减少了模型参数数量，还提高了数据利用率，有助于缓解多智能体学习中的稀疏奖励问题。

### 价值估计优化

#### 广义优势估计（GAE）

使用 GAE 估计优势函数：

$$
A_t^{GAE} = \sum_{l=0}^{T - t} (\gamma\lambda)^l \delta_{t + l}, \text{ 其中 } \delta_k = r_k + \gamma V(s_{k+1}) - V(s_k) ,
$$

并通过优势归一化进一步提高训练稳定性。

#### 价值归一化

针对多智能体场景中因大奖励波动导致的价值学习不稳定问题（例如，SMAC 中胜利与失败的奖励差异可达 200 以上），MAPPO 使用运行均值和标准差对价值目标进行归一化：$V_{\text{norm}}(s_t) = \frac{V(s_t) - \mu_{\text{running}}}{\sigma_{\text{running}} + \epsilon}$

其中 $\mu_{\text{running}}$ 和 $\sigma_{\text{running}}$ 是训练期间实时更新的价值目标的运行均值和标准差，$\epsilon = 1e-8$ 避免除以零。论文的实证结果表明，这种优化可以将复杂 SMAC 地图（例如，3s5z）的胜率提高 15%-20%。

### 裁剪机制

MAPPO 采用 PPO 的双重裁剪机制，分别作用于策略比率和价值损失：

策略裁剪：

$$
\mathcal{L}^{CLIP} = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

其中 $r_t(\theta) = \frac{\pi_{\theta}(a_t|o_t)}{\pi_{\theta_{old}}(a_t|o_t)}$。

价值裁剪：防止价值函数更新过度并提高训练稳定性。

## 算法

MAPPO 的完整训练算法如算法 1 所示：

```{eval-rst}
.. image:: ./../../../_static/figures/pseucodes/pseucode-MAPPO.png
    :width: 80%
```  

## 在 XuanCe 中运行 MAPPO

在 XuanCe 中运行 MAPPO 之前，您需要准备 conda 环境并按照 
[**安装步骤**](./../../usage/installation.rst#install-xuance) 安装 ``xuance``。

### 运行内置演示

完成安装后，您可以打开 Python 控制台并使用以下命令直接运行 MAPPO：

```python3
import xuance
runner = xuance.get_runner(method='mappo',
                    env='mpe',  
                    env_id='simple_spread_v3',  
                    is_test=False)
runner.run() 
```
### 使用自定义配置运行

如果您想使用不同的配置运行 MAPPO，可以创建一个新的 ``.yaml`` 文件，例如 ``my_config.yaml``。
然后，通过以下代码块运行 MAPPO：

```python3
import xuance
runner = xuance.get_runner(method='mappo',
                       env='mpe', 
                       env_id='simple_spread_v3',  
                       config_path="my_config.yaml",  # my_config.yaml 文件的路径应该是正确的。
                       is_test=False)
runner.run()
```
### 使用自定义环境运行

如果您想在 XuanCe 中未包含的自己的环境中运行 XuanCe 的 MAPPO， 
您需要按照
[**新环境教程**](./../../usage/custom_env/custom_drl_env.rst) 中的步骤定义新环境。
然后，[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``mappo_myenv.yaml``。

之后，您可以使用以下代码在自己的环境中运行 MAPPO：

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import MAPPO_Agents

configs_dict = get_configs(file_dir="mappo_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs) 
Agent = MAPPO_Agents(config=configs, envs=envs) 
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

