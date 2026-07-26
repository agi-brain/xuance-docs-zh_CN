# 优势执行-评价算法（A2C）

优势执行—评价算法（Advantage Actor Critic，A2C）是一种结合策略梯度方法与价值函数近似的强化学习算法。它使用优势函数代替动作价值函数，以提升学习过程的稳定性和性能。

下表列出了 A2C 算法的一些基本特征：

| A2C 的特征           | 是否具备 | 说明 |
|-------------------|----------|------|
| 同策略（On-policy）    | ✅ | 评估策略与目标策略相同。 |
| 异策略（Off-policy）   | ❌ | 评估策略与目标策略不同。 |
| 无模型（Model-free）   | ✅ | 无须预先构建环境动力学模型。 |
| 基于模型（Model-based） | ❌ | 需要使用环境模型训练策略。 |
| 离散动作              | ✅ | 可处理离散动作空间。 |
| 连续动作              | ✅ | 可处理连续动作空间。 |

## 执行—评价（AC）框架

执行—评价方法通过执行器（Actor）选择动作，通过评价器（Critic）评估这些动作，并通过二者的协同训练不断改进策略。

### 评价器

评价器也称为价值网络，它使用神经网络 $Q^\pi(s,a;w)$ 近似动作价值函数 $Q^\pi(s,a)$。在单步 Q-Learning 中，通过迭代最小化一系列损失函数来学习动作价值函数 $Q^\pi(s,a;w)$ 的参数 $w$，其中第 $i$ 次迭代的损失函数定义为：

$$
L_i(w_i)=\mathbb{E}[(r+\gamma Q(s',a';w_{i})-Q(s,a;w_i))^2]
$$

其中，$s'$ 表示智能体在状态 $s$ 之后到达的下一状态。

### 执行器

执行器也称为策略网络，其思想与[**策略梯度（PG）**](./pg.md)方法类似。执行器直接优化策略，以最大化累计回报。其目标函数表示为：

$$
J(\theta) = \mathbb{E}_{\pi_{\theta}}{[\sum_{t=0}^{\infty}{\gamma^t r_t}]}.
$$

为了优化策略函数 $\pi_\theta$，需要计算目标函数 $J(\theta)$ 关于参数 $\theta$ 的梯度：

$$
\nabla_{\theta}J(\theta) = \mathbb{E}_{\pi_{\theta}}[\nabla_{\theta}\log{\pi_{\theta}(a|s)Q^{\pi_{\theta}}(s, a)}].
$$

在训练执行器的过程中，使用评价器得到的动作价值 $Q(s,a;w)$ 近似真实的 $Q(s,a)$。通过交替训练执行器和评价器，最终实现最大化 $J(\theta)$ 的目标。

## 优势执行—评价算法（A2C）

在上述执行—评价框架中，使用 $Q(s,a;w)$ 更新策略。在优势执行—评价算法（A2C）中，使用优势函数表示相对于当前状态平均价值而言，执行某个动作所获得的额外收益：

$$
A(a_t,s_t)=Q(a_t,s_t)-V(s_t)\approx r_t+\gamma V(s_{t+1}) - V(s_t),
$$

其中：

$$
Q_\pi(a_t,s_t)-V_\pi(s_t)=\mathbb{E}[R_t+\gamma v_\pi(S_{t+1}) - v_\pi(S_t)|S_t=s_t].
$$

优势函数能够减小策略梯度估计的方差。因此，目标函数 $J(\theta)$ 的梯度可以改写为：

$$
\nabla_{\theta}J(\theta) = \mathbb{E}_{\pi_{\theta}}[\nabla_{\theta}\log{\pi_{\theta}(a|s)A^\pi(s,a)}].
$$

对于评价器网络，其损失函数可以改写为：

$$
L(w)=\mathbb{E}[(r_t+\gamma V(s_{t+1};w)-V(s_t;w))^2].
$$

## 算法框架

XuanCe 中实现的 A2C 结构框架如下图所示。

```{eval-rst}
.. image:: ./../../../_static/figures/algo_framework/a2c_framework.png
    :width: 80%
    :align: center
```

## 在 XuanCe 中运行 A2C

在 XuanCe 中运行 A2C 之前，需要先准备一个 conda 环境，并按照[**安装步骤**](./../../usage/installation.rst)安装 ``xuance``。

### 运行内置示例

完成安装后，可以打开 Python 控制台，并使用以下命令直接运行 A2C：

```python3
import xuance
runner = xuance.get_runner(method='a2c',
                           env='classic_control',  # 可选项：classic_control、box2d、atari。
                           env_id='CartPole-v1',  # 可选项：CartPole-v1、LunarLander-v2、ALE/Breakout-v5 等。
                           is_test=False)
runner.run()  # 也可以使用 runner.benchmark()
```

### 使用自定义配置运行

如需使用不同配置运行 A2C，可以新建一个 ``.yaml`` 文件，例如 ``my_config.yaml``。
然后使用以下代码运行 A2C：

```python3
import xuance as xp
runner = xp.get_runner(method='a2c',
                       env='classic_control',  # 可选项：classic_control、box2d、atari。
                       env_id='CartPole-v1',  # 可选项：CartPole-v1、LunarLander-v2、ALE/Breakout-v5 等。
                       config_path="my_config.yaml",  # 请确保 my_config.yaml 文件的路径正确。
                       is_test=False)
runner.run()  # 也可以使用 runner.benchmark()
```

如需进一步了解配置方法，请参阅[**配置教程**](./../../api/configs/configuration_examples.rst)。

### 在自定义环境中运行

如需在 XuanCe 尚未包含的自定义环境中运行 A2C，需要按照[**新环境教程**](./../../usage/custom_env/custom_drl_env.rst)中的步骤定义新环境。然后，[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations)
``a2c_myenv.yaml``。

完成上述操作后，可以使用以下代码在自定义环境中运行 A2C：

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import A2C_Agent

configs_dict = get_configs(file_dir="a2c_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # 创建并行环境。
Agent = A2C_Agent(config=configs, envs=envs)  # 创建一个来自 XuanCe 的 A2C 智能体。
Agent.train(configs.running_steps // configs.parallels)  # 对模型进行多个步骤的训练。
Agent.save_model("final_train_model.pth")  # 将模型保存到 model_dir。
Agent.finish()  # 结束训练。
```
