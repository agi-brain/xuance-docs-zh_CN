# CURL: Contrastive Unsupervised Representations for Reinforcement Learning
**论文链接:** [**Arxiv**](https://arxiv.org/abs/1807.03748)

Contrastive Unsupervised Representations for Reinforcement Learning（简称 CURL）是一种高样本效率的无模型深度强化学习（Deep Reinforcement Learning, DRL）智能体。它通过对同一状态的多种数据增强版本进行对比学习，从而学习状态表示。CURL 使用的对比损失与计算机视觉中自监督学习方法 [**Contrastive Predictive Coding (CPC)**](https://arxiv.org/abs/1807.03748) 类似。  
学习得到的表示随后被标准的 DQN 智能体用于策略学习。

CURL 由两个主要组件组成：
1. 一个卷积神经网络（CNN）编码器，用于将观测值编码为表示；
2. 一个利用这些表示进行策略学习的 DQN 智能体。

CURL 通过对比同一状态的多种数据增强版本来学习表示。该对比损失鼓励编码器学习对数据增强不敏感（即具有不变性）的表示，同时保留对控制任务至关重要的信息。

CURL 的关键思想在于：它能够在**不依赖动作或奖励**的情况下进行表示学习，因此属于**无监督表示学习方法**。这在奖励稀疏或延迟的环境中特别有用。

---

## 核心组成部分（Core Components）

CURL 主要包含两个部分：

### Encoder（编码器）

编码器是一个卷积神经网络（CNN），用于将观测值映射为高维表示。  
其训练目标是使用对比损失（contrastive loss），使得模型在对抗数据增强的同时保持对控制任务有效的信息。

### DQN Agent（DQN 智能体）

DQN 智能体使用编码器学习得到的表示来进行策略学习。  
该部分本质上是一个标准的 DQN 智能体，通过深度神经网络近似 Q 函数。

---

## Contrastive Loss (InfoNCE)

CURL 使用与 Contrastive Predictive Coding (CPC) 类似的对比损失函数 InfoNCE。  
该损失函数旨在鼓励编码器学习既对增强不敏感又保留控制信息的状态表示。

其定义如下：

$$
\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}_{x \sim \mathcal{D}} \left[ \log \frac{\exp(\text{sim}(q, k^+) / \tau)}{\sum_{k^-} \exp(\text{sim}(q, k^-) / \tau)} \right]
$$

其中：
- $q$ 表示查询（query）表示（编码后的增强观测）
- $k^+$ 表示正样本（即同一观测的另一种增强版本）
- $k^-$ 表示负样本（来自不同状态的增强观测）
- $\tau$ 是温度参数（temperature）
- $\text{sim}(u, v)$ 表示 $u$ 与 $v$ 的余弦相似度

---

## Q-Learning with Contrastive Representations

在利用对比损失学习到状态表示后，CURL 使用标准的 DQN 智能体进行策略学习。  
DQN 网络使用编码器输出的表示来近似 Q 函数。

Q 网络通过最小化预测 Q 值与目标 Q 值的均方误差（MSE）损失进行训练：

$$
L = \mathbb{E}_{(s, a, s', r) \sim \mathcal{D}}[(y - Q(s, a; \theta))^2],
$$

其中  
$y = r + \gamma \max_{a'}{Q(s', a'; \theta^{-})}$，$\theta^{-}$ 为目标网络参数。

CURL 使用 $\epsilon$-贪婪策略（epsilon-greedy policy）在探索与利用之间取得平衡：

$$
\pi(s) = 
\begin{cases}
\arg\max_{a}Q(s, a) & \text{以概率 } 1-\epsilon \text{ 选择最优动作}, \\
\text{随机选择一个动作} & \text{以概率 } \epsilon.
\end{cases}
$$

---

## 超参数（Hyperparameters）

CURL 的主要超参数包括：

- `temperature`: InfoNCE 损失中的温度参数（默认 1.0）  
- `tau`: 用于目标编码器的动量更新系数（默认 0.05）  
- `repr_lr`: 表示学习阶段的学习率（默认 0.0001）  
- `sync_frequency`: 目标网络同步频率（默认 100）

---

## 算法（Algorithm）

CURL 的完整训练流程见算法 1：

```{eval-rst}
.. image:: ./../../../../_static/figures/pseucodes/curl-pytorch.png
    :width: 80%
    :align: center
```

---

## 框架（Framework）

CURL 在 XuanCe 平台中的总体智能体–环境交互框架如下图所示：

```{eval-rst}
.. image:: ./../../../../_static/figures/algo_framework/curl_framework.png
    :width: 100%
    :align: center
```

---

## 在 XuanCe 中运行 CURL（Run CURL in XuanCe）

在运行 CURL 之前，需要准备 Conda 环境并按照  
[**安装步骤**](./../../../usage/installation.rst#install-xuance) 安装 ``xuance``。

### 运行内置示例（Run Built-in Demos）

安装完成后，可在 Python 控制台中直接运行：

```python3
import xuance
runner = xuance.get_runner(method='curl',
                           env='atari',
                           env_id='ALE/Breakout-v5',
                           is_test=False)
runner.run()  # 或 runner.benchmark()
```

---

### 使用自定义配置文件（Run With Self-defined Configs）

若希望使用自定义配置运行 CURL，可新建 ``.yaml`` 文件（例如 ``my_config.yaml``），  
然后执行以下代码：

```python3
import xuance as xp
runner = xp.get_runner(method='curl',
                       env='atari',
                       env_id='ALE/Breakout-v5',
                       config_path="my_config.yaml",
                       is_test=False)
runner.run()  # 或 runner.benchmark()
```

更多关于配置文件的内容可参见  
[**配置教程**](./../../configs/configuration_examples.rst)。

---

### 在自定义环境中运行（Run With Customized Environment）

若希望在自定义环境中运行 CURL，需要先按照  
[**新环境教程**](./../../../usage/new_envs.rst) 定义新的环境，  
并 [**准备配置文件**](./../../../usage/new_envs.rst#step-2-create-the-config-file-and-read-the-configurations) ``curl_myenv.yaml``。

之后可使用以下代码运行：

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import CURL_Agent

configs_dict = get_configs(file_dir="curl_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)
Agent = CURL_Agent(config=configs, envs=envs)
Agent.train(configs.running_steps // configs.parallels)
Agent.save_model("final_train_model.pth")
Agent.finish()
```

---

## 引用（Citation）

```bash
@inproceedings{laskin2020curl,
  title={Curl: Contrastive unsupervised representations for reinforcement learning},
  author={Laskin, Michael and Srinivas, Aravind and Abbeel, Pieter},
  booktitle={International Conference on Machine Learning},
  pages={5639--5650},
  year={2020},
  organization={PMLR}
}
```

---

## API 接口（APIs）

### PyTorch

```{eval-rst}
.. automodule:: xuance.torch.agents.contrastive_unsupervised_rl.curl_agent
    :members:
    :undoc-members:
    :show-inheritance:
```

### TensorFlow2

```{eval-rst}
.. automodule:: xuance.tensorflow.agents.contrastive_unsupervised_rl.curl_agent
    :members:
    :undoc-members:
    :show-inheritance:
```

### MindSpore

```{eval-rst}
.. automodule:: xuance.mindspore.agents.contrastive_unsupervised_rl.curl_agent
    :members:
    :undoc-members:
    :show-inheritance:
```
