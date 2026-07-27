# 分离式参数化深度 Q 网络（SP-DQN）

**算法全称**：**Split Parameterized Deep Q-Network (SP-DQN)**

分离式参数化深度 Q 网络（Split Parameterized Deep Q-Network，SP-DQN）是传统深度 Q 网络（DQN）的一种扩展，旨在提高 Q-learning 在大规模问题中的计算效率和可扩展性。SP-DQN 将 Q 网络拆分为多个参数化子网络，每个子网络对应不同的动作空间。通过对子网络进行解耦，SP-DQN 可以在保持较好性能的同时降低训练复杂度和内存需求。

下表列出了 SP-DQN 算法的一些基本特征：

| SP-DQN 的特征        | 是否具备   | 说明             |
|-------------------|--------|----------------|
| 同策略（On-policy）    | ❌      | 评估策略与目标策略相同。   |
| 异策略（Off-policy）   | ✅      | 评估策略与目标策略不同。   |
| 无模型（Model-free）   | ✅      | 无须预先构建环境动力学模型。 |
| 基于模型（Model-based） | ❌      | 需要使用环境模型训练策略。  |
| 离散动作              | ✅      | 可处理离散动作空间。     |
| 连续动作              | ❌      | 不处理连续动作空间。     |

## Q-learning 回顾

[**Q-learning**](https://link.springer.com/article/10.1007/bf00992698) 是一种无模型强化学习算法。智能体学习动作价值函数 $Q(s,a)$，用于估计在状态 $s$ 下执行动作 $a$，并在之后遵循最优策略时所能获得的期望累积回报。

Q-learning 的 [**Bellman 方程**](https://en.wikipedia.org/wiki/Bellman_equation) 为：

$$
Q(s,a)
\leftarrow
Q(s,a)
+
\alpha
\left[
r+
\gamma
\max_{a'}Q(s',a')
-
Q(s,a)
\right].
$$

其中，$\alpha$ 表示学习率，$r$ 表示奖励，$\gamma$ 表示折扣因子，$s'$ 表示下一状态。

## 分离式参数化 Q-learning

SP-DQN 的核心思想是将 Q 网络拆分为多个参数化部分。每个部分负责估计特定动作子集对应的 Q 值。由于不同部分可以相互独立或并行训练，这种结构能够降低训练过程中的计算复杂度。

SP-DQN 将参数拆分机制应用于 Q 函数，使模型在具有大规模动作空间的环境中能够更加高效地表示和计算动作价值。

## 使用分离参数的深度 Q 网络

SP-DQN 通过使用分离式网络结构对 DQN 进行改进。神经网络被划分为多个部分，每个部分对应一种特定的动作类别。其一般步骤如下：

1. **动作空间拆分：** 将动作空间划分为彼此不重叠的子集，并为每个子集分配一个独立的参数化网络。
2. **独立网络训练：** 分别训练各个参数化网络，从而降低训练复杂度和内存占用。
3. **合并 Q 函数：** 将各个独立网络的输出组合起来，得到每个动作对应的最终 Q 值。

训练 SP-DQN 所使用的损失函数与 DQN 类似：

$$
L
=
\mathbb{E}_{(s,a,s',r)\sim\mathcal{D}}
\left[
\left(
y-Q(s,a;\theta)
\right)^2
\right],
$$

其中：

$$
y
=
r+
\gamma
\max_{a'}
Q(s',a';\theta^-),
$$

$\theta^-$ 表示目标网络的参数。

## $\epsilon$-贪心探索

SP-DQN 使用与 DQN 相同的 $\epsilon$-贪心探索策略：

$$
\pi(s)
=
\begin{cases}
\arg\max_a Q(s,a),
& \text{以概率 }1-\epsilon,\\
\text{随机动作},
& \text{以概率 }\epsilon.
\end{cases}
$$

该策略使智能体以概率 $\epsilon$ 随机探索环境，并以概率 $1-\epsilon$ 利用当前已经学习到的策略。

## 算法

训练 SP-DQN 的完整算法如算法 1 所示。

## 算法框架

XuanCe 中实现的 SP-DQN 智能体与环境交互框架如下图所示。

## 在 XuanCe 中运行 SP-DQN

在 XuanCe 中运行 SP-DQN 之前，需要先准备一个 conda 环境，并按照
[**安装步骤**](./../../usage/installation.rst)
安装 ``xuance``。

### 运行内置示例

完成安装后，可以打开 Python 控制台，并使用以下命令直接运行 SP-DQN：

```python3
import xuance

runner = xuance.get_runner(
    algo='spdqn',
    env='classic_control',  # 可选项：classic_control、box2d、atari。
    env_id='CartPole-v1',  # 可选项：CartPole-v1、LunarLander-v3、ALE/Breakout-v5 等。
)
runner.run()  # 也可以使用 runner.benchmark()
```

### 使用自定义配置运行

如需使用不同配置运行 SP-DQN，可以新建一个 ``.yaml`` 文件，例如
``my_config.yaml``。然后使用以下代码运行 SP-DQN：

```python3
import xuance as xp

runner = xp.get_runner(
    algo='spdqn',
    env='classic_control',  # 可选项：classic_control、box2d、atari。
    env_id='CartPole-v1',  # 可选项：CartPole-v1、LunarLander-v3、ALE/Breakout-v5 等。
    config_path="my_config.yaml",  # 请确保 my_config.yaml 文件的路径正确。
)
runner.run()  # 也可以使用 runner.benchmark()
```

### 在自定义环境中运行

如需在 XuanCe 尚未包含的自定义环境中运行 SP-DQN，需要按照
[**新环境教程**](./../../usage/custom_env/custom_drl_env.rst)
中的步骤定义新环境。然后，
[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations)
``spdqn_myenv.yaml``。

完成上述操作后，可以使用以下代码运行 SP-DQN：

```python3
import argparse
from xuance.common import load_yaml
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import SP_DQN_Agent

configs_dict = load_yaml(file_dir="spdqn_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # 创建并行环境。
Agent = SP_DQN_Agent(config=configs, envs=envs)  # 创建 SP-DQN 智能体。
Agent.train(configs.running_steps // configs.parallels)  # 训练模型。
Agent.save_model("final_train_model.pth")  # 将模型保存到 model_dir。
Agent.finish()  # 结束训练。
```

## 参考文献

```{code-block} bash
@article{he2017split,
  title={Split Parameterized Deep Q-Network},
  author={He, Xun and Zhang, Xiang and Liu, Jian and Lu, Yun},
  journal={arXiv preprint arXiv:1707.02785},
  year={2017}
}
```
