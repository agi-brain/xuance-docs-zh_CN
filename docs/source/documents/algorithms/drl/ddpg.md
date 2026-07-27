# 深度确定性策略梯度（DDPG）

**论文链接：** [**https://arxiv.org/abs/1509.02971**](https://arxiv.org/abs/1509.02971)。

深度确定性策略梯度（Deep Deterministic Policy Gradient，DDPG）是一种无模型深度强化学习算法，它将策略梯度方法与深度神经网络相结合。该算法由 Timothy P. Lillicrap 等人于 2015 年提出。DDPG 已被广泛应用于连续控制任务，并在机器人控制和仿真环境等场景中取得了显著成果。

下表列出了 DDPG 算法的一些基本特征：

| DDPG 的特征 | 是否具备 | 说明 |
|--------------|----------|------|
| 同策略（On-policy） | ❌ | 评估策略与目标策略相同。 |
| 异策略（Off-policy） | ✅ | 评估策略与目标策略不同。 |
| 无模型（Model-free） | ✅ | 无须预先构建环境动力学模型。 |
| 基于模型（Model-based） | ❌ | 需要使用环境模型训练策略。 |
| 离散动作 | ❌ | 不直接处理离散动作空间。 |
| 连续动作 | ✅ | 可处理连续动作空间。 |

## 演员—评论家框架

DDPG 建立在演员—评论家（Actor-Critic）框架之上。演员网络负责根据当前状态生成动作，并通过深度神经网络直接输出确定性动作。评论家网络则用于估计状态—动作对的 Q 值。评论家网络同时接收状态和动作作为输入，并输出相应的 Q 值估计，该值表示在给定状态下执行特定动作所能获得的长期期望回报。

## 策略梯度

DDPG 中的演员网络使用策略梯度方法进行更新。策略梯度通过计算期望回报关于演员网络参数的梯度得到，其目标是寻找能够最大化期望回报的策略。演员网络的更新规则基于 Q 值关于演员参数的梯度，并通过链式法则计算。

演员网络参数 $\theta^\mu$ 的梯度为：

$$
\nabla_{\theta^\mu} J \approx \frac{1}{N} \sum_{i=1}^{N} \nabla_a Q(s, a \mid \theta^Q) \bigg|_{s=s^i, a=\mu(s^i \mid \theta^\mu)} \nabla_{\theta^\mu} \mu(s \mid \theta^\mu) \bigg|_{s^i}
$$

其中，$J$ 表示期望回报，$N$ 表示一个小批量中的样本数量，$\theta^Q$ 表示评论家网络的参数，$\mu(s \mid \theta^\mu)$ 表示演员网络。

## 评论家网络更新

DDPG 中的评论家网络使用时序差分（Temporal-Difference，TD）误差进行更新。TD 误差是目标 Q 值与预测 Q 值之间的差。目标 Q 值使用与 Q-learning 类似的 Bellman 方程计算。评论家网络通过最小化预测 Q 值和目标 Q 值之间的均方误差（Mean-Squared Error，MSE）进行更新。

对于样本 $(s^i, a^i, r^i, s^{i+1})$，其目标 Q 值 $y^i$ 为：

$$
y^i = r^i + \gamma Q'(s^{i+1}, \mu'(s^{i+1} \mid \theta^{\mu'}) \mid \theta^{Q'})
$$

其中，$\mu'$ 和 $Q'$ 分别表示目标演员网络和目标评论家网络，$\theta^{\mu'}$ 和 $\theta^{Q'}$ 分别表示二者的参数，$\gamma$ 表示折扣因子。

评论家网络参数 $\theta^Q$ 通过最小化以下损失函数进行更新：

$$
L = \frac{1}{N} \sum_{i=1}^{N} (y^i - Q(s^i, a^i \mid \theta^Q))^2
$$

## 目标网络与经验回放

与 DQN 类似，DDPG 使用目标网络稳定学习过程。算法分别维护目标演员网络和目标评论家网络，并采用软更新规则，根据主网络的参数缓慢更新目标网络：

$$
\theta' \leftarrow \tau \theta + (1 - \tau) \theta'
$$

其中，$\tau$ 是一个较小的正数，通常取 0.005 左右，用于控制目标网络的更新速度。

DDPG 还使用经验回放机制。经验回放缓冲区用于存储智能体的交互经验 $(s^i, a^i, r^i, s^{i+1})$。训练演员网络和评论家网络时，从缓冲区中随机采样小批量经验。这有助于减弱连续样本之间的相关性，并提高学习算法的稳定性和泛化能力。

## 探索机制

DDPG 使用探索策略鼓励智能体探索环境。通常会在演员网络生成的动作上添加噪声过程：

$$
\mu'(s_t) = \mu(s_t \mid \theta^{\mu}_t) + \mathcal{N}
$$

噪声 $\mathcal{N}$ 可以采用高斯噪声等形式。探索噪声能够帮助智能体探索动作空间中的不同区域，并发现更优的策略。随着训练过程推进，探索噪声的幅度可以逐渐减小。

DDPG 的主要优点包括：

- 能够处理连续动作空间，适用于机器人控制等多种连续控制任务。
- 通过目标网络和经验回放机制稳定学习过程，提高训练的可靠性。
- 已在机器人操作和运动控制等多种连续控制场景中展现出良好性能。

## 算法

训练 DDPG 的完整算法如算法 1 所示：

```{eval-rst}
.. image:: ./../../../_static/figures/pseucodes/pseucode-DDPG.png
    :width: 80%
    :align: center
```

## 在 XuanCe 中运行 DDPG

在 XuanCe 中运行 DDPG 之前，需要先准备一个 conda 环境，并按照
[**安装步骤**](./../../usage/installation.rst)安装 ``xuance``。

### 运行内置示例

完成安装后，可以打开 Python 控制台，并使用以下命令直接运行 DDPG：

```python3
import xuance
runner = xuance.get_runner(method='ddpg',
                           env='classic_control',  # 可选项：classic_control、box2d 等。
                           env_id='Pendulum-v1',  # 选择具有连续动作空间的环境。
                           is_test=False)
runner.run()  # 也可以使用 runner.benchmark()
```

### 使用自定义配置运行

如需使用不同配置运行 DDPG，可以新建一个 ``.yaml`` 文件，例如 ``my_config.yaml``。
然后使用以下代码运行 DDPG：

```python3
import xuance as xp
runner = xp.get_runner(method='ddpg',
                       env='classic_control',  # 可选项：classic_control、box2d 等。
                       env_id='Pendulum-v1',  # 选择具有连续动作空间的环境。
                       config_path="my_config.yaml",  # 请确保 my_config.yaml 文件的路径正确。
                       is_test=False)
runner.run()  # 也可以使用 runner.benchmark()
```

如需进一步了解配置方法，请参阅
[**配置教程**](./../../api/configs/configuration_examples.rst)。

### 在自定义环境中运行

如需在 XuanCe 尚未包含的自定义环境中运行 DDPG，
需要按照[**新环境教程**](./../../usage/custom_env/custom_drl_env.rst)
中的步骤定义新环境。
然后，[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations)
``ddpg_myenv.yaml``。

完成上述操作后，可以使用以下代码在自定义环境中运行 DDPG：

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import DDPG_Agent

configs_dict = get_configs(file_dir="ddpg_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # 创建并行环境。
Agent = DDPG_Agent(config=configs, envs=envs)  # 创建一个来自 XuanCe 的 DDPG 智能体。
Agent.train(configs.running_steps // configs.parallels)  # 对模型进行多个步骤的训练。
Agent.save_model("final_train_model.pth")  # 将模型保存到 model_dir。
Agent.finish()  # 结束训练。
```

## 参考文献

```{code-block} bash
@article{lillicrap2015continuous,
  title={Continuous control with deep reinforcement learning},
  author={Lillicrap, Timothy P and Hunt, Jonathan J and Pritzel, Alexander and Heess, Nicolas and Erez, Tom and Tassa, Yuval and Silver, David and Wierstra, Daan},
  journal={arXiv preprint arXiv:1509.02971},
  year={2015}
}
```
