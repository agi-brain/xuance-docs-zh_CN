# DQN with Noisy Layers (Noisy DQN)

**论文链接:** [**https://arxiv.org/pdf/1706.01905**](https://arxiv.org/pdf/1706.01905).

Noisy DQN 是传统深度Q网络（DQN）的一种变体，它在Q网络的权重中引入了噪声，以改善学习过程中的探索能力。
此举旨在解决强化学习中的关键挑战之一：平衡探索（exploration）与利用（exploitation）。

下表列出了 Noisy DQN 算法的一些一般特性:

| Noisy DQN 特性            | 值 | 描述            |
|-------------------------|---|---------------|
| 同轨策略（On-policy）         | ❌ | 评估策略与目标策略相同。  |
| 离轨策略（Off-policy）        | ✅ | 评估策略与目标策略不同。  | 
| 无模型（Model-free）         | ✅ | 不需要准备环境动态模型。  | 
| 有模型（Model-based）        | ❌ | 需要环境模型来训练策略。  | 
| 离散动作（Discrete Action）   | ✅ | 处理离散动作空间。     |   
| 连续动作（Continuous Action） | ❌ | 处理连续动作空间。     |

## Noisy DQN 的核心理念

**探索与利用 (Exploration vs. Exploitation)**：
在标准的 DQN 中，探索通常由 $\epsilon$-贪婪策略控制，其中智能体以一定的概率（epsilon）随机选择动作，并在其余时间利用已知最优的动作。
Noisy DQN 试图通过直接将噪声引入网络参数中来解决探索难题，而不是仅仅依赖随机的动作选择。

**噪声网络 (Noisy Networks)**：
Noisy DQN 并没有使用固定的 epsilon 进行探索，而是将噪声引入到了Q网络本身的参数中。
这是通过向Q网络的权重添加参数噪声来实现的，这会改变输出的Q值，从而鼓励去探索不同的动作和状态。

**噪声线性层 (Noisy Linear Layers)**：
在 Noisy DQN 架构中，神经网络传统的全连接层被替换为“噪声”层。
这些噪声层在训练期间向层的权重添加噪声，使得智能体的决策过程在本质上更具探索性。

**噪声网络公式 (The Noisy Network Formula)**：对于网络中的每一层，权重被参数化为：

$$
w = \mu + \sigma \cdot \epsilon,
$$

其中:
- $\mu$ 是均值或基础权重;
- $\sigma$ 是控制噪声水平的标准差;
- $\epsilon$ 是从噪声分布（通常为高斯分布）中提取的样本. 
噪声 $\epsilon$ 在每个回合（episode）或每次迭代开始时进行采样，以确保噪声在训练期间是动态的。

Noisy DQN 具有三个主要优点:

- **改善探索能力**: 通过在Q值中引入噪声，智能体被鼓励去探索更广泛的动作，而不是仅仅利用当前已知的最佳动作.
- **自适应探索**: 探索水平可以作为训练的一部分自动调整，从而消除了手动调整探索参数（如 epsilon）的需要.
- **高效训练**: Noisy DQN 可以提高样本效率，因为它利用探索去访问较少遇到的状态，从而有可能在复杂环境中获得更好的性能.

## 框架

Noisy DQN 保留了与 
[**DQN**](dqn.md#framework) 
相同的整体结构（即经验回放、目标网络等），但用Q网络中的噪声层取代了原有的探索机制。

## 在 XuanCe 中运行 Noisy DQN

在 XuanCe 中运行 Noisy DQN 之前，您需要准备一个 conda 环境并按照[**安装步骤**](./../../usage/installation.rst).

### 运行内置案例

完成安装后，您可以打开 Python 控制台并使用以下命令直接运行 Noisy DQN:

```python3
import xuance
runner = xuance.get_runner(method='noisydqn',
                           env='classic_control',  # Choices: claasi_control, box2d, atari.
                           env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                           is_test=False)
runner.run()  # Or runner.benchmark()
```

### 使用自定义配置运行

如果您想使用不同的配置运行 Noisy DQN，您可以创建一个新的 ``.yaml`` 文件，例如 ``my_config.yaml``。
然后，通过以下代码块运行 Noisy DQN：

```python3
import xuance as xp
runner = xp.get_runner(method='noisydqn',
                       env='classic_control',  # Choices: claasi_control, box2d, atari.
                       env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)
runner.run()  # Or runner.benchmark()
```

要了解有关配置的更多信息，请访问
[**配置教程**](./../../api/configs/configuration_examples.rst).

### 使用自定义环境运行

如果您想在未包含于 XuanCe 的自有环境中运行 XuanCe 的 Noisy DQN，您需要按照 
[**新环境教程**](./../../usage/custom_env/custom_drl_env.rst)。
然后，[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
``noisydqn_myenv.yaml``.

之后，您可以使用以下代码在自己的环境中运行 Noisy DQN:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import NoisyDQN_Agent

configs_dict = get_configs(file_dir="noisydqn_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = NoisyDQN_Agent(config=configs, envs=envs)  # Create a DDPG agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```

## 引用

```{code-block} bash
@inproceedings{
  plappert2018parameter,
  title={Parameter Space Noise for Exploration},
  author={Matthias Plappert and Rein Houthooft and Prafulla Dhariwal and Szymon Sidor and Richard Y. Chen and Xi Chen and Tamim Asfour and Pieter Abbeel and Marcin Andrychowicz},
  booktitle={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=ByBAl2eAZ},
}
```
