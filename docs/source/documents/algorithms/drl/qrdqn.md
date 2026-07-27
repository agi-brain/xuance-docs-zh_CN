# Quantile Regression Deep Q-Network (QR-DQN)

**论文链接：** [**https://ojs.aaai.org/index.php/AAAI/article/view/11791**](https://ojs.aaai.org/index.php/AAAI/article/view/11791)。

分位数回归深度 Q 网络（Quantile Regression Deep Q-Network，QR-DQN）是传统 DQN 的一种扩展，
旨在提升强化学习算法对不确定性和方差的处理能力，
尤其适用于奖励具有较大波动或包含较强噪声的环境。
QR-DQN 将分位数回归与 DQN 相结合，
使其能够学习 Q 值的概率分布，而不仅仅是单个点估计。
这有助于提高学习过程的稳定性和鲁棒性。

下表列出了 QR-DQN 算法的一些基本特征：

| QR-DQN 的特征        | 是否具备 | 说明             |
|-------------------|------|----------------|
| 同策略（On-policy）    | ❌    | 评估策略与目标策略相同。   |
| 异策略（Off-policy）   | ✅    | 评估策略与目标策略不同。   |
| 无模型（Model-free）   | ✅    | 无须预先构建环境动力学模型。 |
| 基于模型（Model-based） | ❌    | 需要使用环境模型训练策略。  |
| 离散动作              | ✅    | 可处理离散动作空间。     |
| 连续动作              | ❌    | 可处理连续动作空间。     |

## 方法

### 分布式强化学习

传统 Q-learning 为每个状态—动作对估计期望回报，即回报分布的均值。
然而，在许多情况下，回报具有不确定性或较大的波动性，
仅关注均值可能无法完整描述这种不确定性。

分布式强化学习不仅估计期望值，
还尝试对每个状态—动作对可能产生的完整回报分布进行建模。

### 分位数回归

分位数回归是一种估计概率分布中特定分位数
（例如第 50 百分位数或第 90 百分位数）而非均值的技术。
通过估计多个分位数，模型可以刻画可能回报的整体分布，
从而提供有关未来奖励波动性的更丰富信息。

在 QR-DQN 中，智能体不再只学习单个 Q 值，
而是学习 Q 值分布中的多个分位数。

### QR-DQN 的结构

在 QR-DQN 中，Q 值函数由可能回报的概率分布表示。
具体而言，智能体使用一组分位数值近似回报分布的分位数函数。

分位数 $\tau_i$（其中 $\tau_i \in [0, 1]$）对应回报分布中的不同位置，
例如第 10、第 50 和第 90 百分位数。
该算法通过优化分位数回归损失来估计 Q 值分布的各个分位数，
而不是只学习单个期望 Q 值。

### 损失函数

QR-DQN 使用分位数 Huber 损失。
该损失将对异常值不太敏感的 Huber 损失函数与分位数损失结合起来。
分位数损失根据模型对目标 Q 值分布中指定分位数的预测准确程度进行惩罚。

对于给定分位数 $\tau$，其分位数损失定义为：

$$
L_{\tau}(Q, \hat{Q}) = \rho_{\tau}(r - Q),
$$

其中，$r$ 是目标回报，即实际奖励或下一状态的预测价值；
$Q$ 是给定状态—动作对对应的预测分位数值；
$\hat{Q}$ 是通过 Bellman 备份得到的相应目标分位数；
$\rho_{\tau}(z)$ 是如下定义的检验函数（check function）：

$$
\rho_{\tau}(z) = z(\tau - \mathbb{I}[z<0]),
$$

其中，$\mathbb{I}[z<0]$ 是指示函数：当 $z<0$ 时取值为 1，否则取值为 0。

分位数回归损失促使模型学习合适的分位数值，
从而尽可能减小预测分位数与真实回报分布之间的差异。

## 算法

训练 QR-DQN 的完整算法如算法 1 所示：

```{eval-rst}
.. image:: ./../../../_static/figures/pseucodes/pseucode-QRDQN.png
    :width: 70%
    :align: center
```

## 在 XuanCe 中运行 QR-DQN

在 XuanCe 中运行 QR-DQN 之前，需要先准备一个 conda 环境，并按照
[**安装步骤**](./../../usage/installation.rst)安装 ``xuance``。

### 运行内置示例

完成安装后，可以打开 Python 控制台，并使用以下命令直接运行 QR-DQN：

```python3
import xuance
runner = xuance.get_runner(method='qrdqn',
                           env='classic_control',  # 可选项：claasi_control、box2d、atari。
                           env_id='CartPole-v1',  # 可选项：CartPole-v1、LunarLander-v2、ALE/Breakout-v5 等。
                           is_test=False)
runner.run()  # 也可以使用 runner.benchmark()
```

### 使用自定义配置运行

如需使用不同配置运行 QR-DQN，可以新建一个 ``.yaml`` 文件，例如 ``my_config.yaml``。
然后使用以下代码运行 QR-DQN：

```python3
import xuance as xp
runner = xp.get_runner(method='qrdqn',
                       env='classic_control',  # 可选项：claasi_control、box2d、atari。
                       env_id='CartPole-v1',  # 可选项：CartPole-v1、LunarLander-v2、ALE/Breakout-v5 等。
                       config_path="my_config.yaml",  # 请确保 my_config.yaml 文件的路径正确。
                       is_test=False)
runner.run()  # 也可以使用 runner.benchmark()
```

如需进一步了解配置方法，请参阅
[**配置教程**](./../../api/configs/configuration_examples.rst)。

### 在自定义环境中运行

如需在 XuanCe 尚未包含的自定义环境中运行 QR-DQN，
需要按照 [**新环境教程**](./../../usage/custom_env/custom_drl_env.rst)
中的步骤定义新环境。
然后，[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations)
``qrdqn_myenv.yaml``。

完成上述操作后，可以使用以下代码在自定义环境中运行 QR-DQN：

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import QRDQN_Agent

configs_dict = get_configs(file_dir="qrdqn_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # 创建并行环境。
Agent = QRDQN_Agent(config=configs, envs=envs)  # 创建一个来自 XuanCe 的 QR-DQN 智能体。
Agent.train(configs.running_steps // configs.parallels)  # 对模型进行多个步骤的训练。
Agent.save_model("final_train_model.pth")  # 将模型保存到 model_dir。
Agent.finish()  # 结束训练。
```

## 参考文献

```{code-block} bash
@inproceedings{dabney2018distributional,
  title={Distributional reinforcement learning with quantile regression},
  author={Dabney, Will and Rowland, Mark and Bellemare, Marc and Munos, R{\'e}mi},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={32},
  number={1},
  year={2018}
}
```
