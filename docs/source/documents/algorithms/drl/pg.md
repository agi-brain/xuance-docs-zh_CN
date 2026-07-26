# 策略梯度（PG）

**论文链接：** [**下载 PDF**](https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)

策略梯度（Policy Gradient，PG）算法由
[Richard Sutton](http://www.incompleteideas.net/) 等人在其 1999 年的经典论文
“Policy Gradient Methods for Reinforcement Learning with Function Approximation”
中提出，是强化学习中直接优化策略的一类基础方法。
在基于价值的方法（如 Q-learning）难以有效处理的场景中，
例如高维动作空间或连续动作空间，策略梯度方法通常尤其有效。

| PG 的特征             | 是否具备 | 说明 |
|--------------------|----------|------|
| 同策略（On-policy）     | ✅ | 评估策略与目标策略相同。 |
| 异策略（Off-policy）    | ❌ | 评估策略与目标策略不同。 |
| 无模型（Model-free）    | ✅ | 无须预先构建环境动力学模型。 |
| 基于模型（Model-based）  | ❌ | 需要使用环境模型训练策略。 |
| 离散动作               | ✅ | 可处理离散动作空间。 |
| 连续动作               | ✅ | 可处理连续动作空间。 |

## 方法

### 动机

PG 方法的目标是通过最大化期望累积奖励，
直接优化由参数 $\theta$ 表示的策略 $\pi_{\theta}(a | s)$：

$$
J(\theta) = \mathbb{E}_{\pi_{\theta}}{[\sum_{t=0}^{\infty}{\gamma^t r_t}]}.
$$

PG 方法不再近似价值函数，
而是计算目标函数 $J(\theta)$ 关于策略参数 $\theta$ 的梯度，
然后更新 $\theta$，以最大化 $J(\theta)$。

### 策略梯度

PG 算法的核心是策略梯度定理，其表达式为：

$$
\nabla_{\theta}J(\theta) = \mathbb{E}_{\pi_{\theta}}[\nabla_{\theta}\log{\pi_{\theta}(a|s)Q^{\pi_{\theta}}(s, a)}],
$$

其中：

- $\pi_{\theta}(a|s)$：随机策略，输出在状态 $s$ 下选择动作 $a$ 的概率。
- $Q^{\pi_{\theta}}(s, a)$：当前策略下的动作价值函数。
- $\nabla_{\theta}\log{\pi_{\theta}(a|s)}$：对数策略关于参数 $\theta$ 的梯度，通常称为得分函数（score function）。

该表达式使策略能够沿着期望回报的梯度方向进行更新。

## 在 XuanCe 中运行 PG

在 XuanCe 中运行 PG 之前，需要先准备一个 conda 环境，并按照
[**安装步骤**](./../../usage/installation.rst)安装 ``xuance``。

### 运行内置示例

完成安装后，可以打开 Python 控制台，并使用以下命令直接运行 PG：

```python3
import xuance
runner = xuance.get_runner(method='pg',
                           env='classic_control',  # 可选项：claasi_control、box2d、atari。
                           env_id='CartPole-v1',  # 可选项：CartPole-v1、LunarLander-v2、ALE/Breakout-v5 等。
                           is_test=False)
runner.run()  # 也可以使用 runner.benchmark()
```

### 使用自定义配置运行

如需使用不同配置运行 PG，可以新建一个 ``.yaml`` 文件，例如 ``my_config.yaml``。
然后使用以下代码运行 PG：

```python3
import xuance as xp
runner = xp.get_runner(method='pg',
                       env='classic_control',  # 可选项：claasi_control、box2d、atari。
                       env_id='CartPole-v1',  # 可选项：CartPole-v1、LunarLander-v2、ALE/Breakout-v5 等。
                       config_path="my_config.yaml",  # 请确保 my_config.yaml 文件的路径正确。
                       is_test=False)
runner.run()  # 也可以使用 runner.benchmark()
```

如需进一步了解配置方法，请参阅
[**配置教程**](./../../api/configs/configuration_examples.rst)。

### 在自定义环境中运行

如需在 XuanCe 尚未包含的自定义环境中运行 PG，
需要按照[**新环境教程**](./../../usage/custom_env/custom_drl_env.rst)
中的步骤定义新环境。
然后，[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations)
``pg_myenv.yaml``。

完成上述操作后，可以使用以下代码在自定义环境中运行 PG：

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import PG_Agent

configs_dict = get_configs(file_dir="pg_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # 创建并行环境。
Agent = PG_Agent(config=configs, envs=envs)  # 创建一个来自 XuanCe 的 PG 智能体。
Agent.train(configs.running_steps // configs.parallels)  # 对模型进行多个步骤的训练。
Agent.save_model("final_train_model.pth")  # 将模型保存到 model_dir。
Agent.finish()  # 结束训练。
```

## 参考文献

```{code-block} bash
@article{sutton1999policy,
  title={Policy gradient methods for reinforcement learning with function approximation},
  author={Sutton, Richard S and McAllester, David and Singh, Satinder and Mansour, Yishay},
  journal={Advances in neural information processing systems},
  volume={12},
  year={1999}
}
```
