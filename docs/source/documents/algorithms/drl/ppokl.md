# 基于 KL 散度的近端策略优化（PPO-KL）

**论文链接：** [**arXiv**](https://arxiv.org/pdf/1707.06347)

**近端策略优化（Proximal Policy Optimization，PPO）** 算法由 **Trust Region Policy Optimization（TRPO）** 的第一作者 **John Schulman** 于 2017 年在 OpenAI 提出。当时，他已从加州大学伯克利分校获得博士学位。
与 TRPO 相比，PPO 保留了限制策略更新幅度这一核心思想，但显著降低了实现复杂度。TRPO 的计算复杂度较高，
尤其是在计算 Hessian 矩阵并执行二阶优化时，因此不适合计算资源有限的场景。
PPO 在继承 TRPO 核心思想的同时，提供了更简单的实现方式。
大量实验结果表明，PPO 的学习效果与 TRPO 相当，甚至可能更快，
因此已成为一种非常流行的强化学习算法。

下表列出了 **PPO** 算法的一些基本特征：

| PPO 的特征           | 是否具备 | 说明              |
|-------------------|------|-----------------|
| 同策略（On-policy）    | ✅    | 评估策略与目标策略相同。    |
| 异策略（Off-policy）   | ❌    | 评估策略与目标策略不同。    |
| 无模型（Model-free）   | ✅    | 无须预先构建环境动力学模型。  |
| 基于模型（Model-based） | ❌    | 需要使用环境模型训练策略。   |
| 离散动作              | ✅    | 可处理离散动作空间。      |
| 连续动作              | ✅    | 可处理连续动作空间。      |

*原论文提出了 PPO 的两种主要变体：PPO-KL 和 PPO-Clip。实验发现，PPO-Clip 通常具有更好的性能和稳定性，
因此后续几乎所有 PPO 实现都采用了裁剪代理目标（Clipped Surrogate Objective）。
本节重点介绍 PPO-KL。*

## TRPO

由于 PPO 是在 TRPO 基础上提出的改进方法，
为了更深入地理解 PPO 的原理，
有必要首先分析 TRPO 的核心思想。

TRPO 算法最初由 John Schulman 等人在 2015 年的论文 [**Trust Region Policy Optimization**](https://proceedings.mlr.press/v37/schulman15.pdf) 中提出。
该论文引入了**信赖域和 KL 散度约束**的概念。
其核心思想是在信赖域内更新策略时，为策略性能提供安全性保证。
TRPO 描述了一种迭代式策略优化过程，在理论上能够保证策略学习性能单调提升，
并且在实际应用中取得了优于普通策略梯度算法的效果。

### 目标函数的单调性保证

TRPO 首先分析新旧策略目标函数之间的差异：

$$
\begin{aligned}
J(\theta^{\prime})-J(\theta) & =\mathbb{E}_{\pi}\left[V^{\pi_{\theta^{\prime}}}(s_0)\right]-\mathbb{E}_{\pi}\left[V^{\pi_\theta}(s_0)\right] \\
 & =\mathbb{E}_{\pi_{\theta^{\prime}}}\left[\sum_{t=0}^\infty\gamma^tr(s_t,a_t)\right]+\mathbb{E}_{\pi_{\theta^{\prime}}}\left[\sum_{t=0}^\infty\gamma^t\left(\gamma V^{\pi_\theta}(s_{t+1})-V^{\pi_\theta}(s_t)\right)\right] \\
 & =\mathbb{E}_{\pi_{\theta^{\prime}}}\left[\sum_{t=0}^\infty\gamma^t\left[r(s_t,a_t)+\gamma V^{\pi_\theta}(s_{t+1})-V^{\pi_\theta}(s_t)\right]\right]
\end{aligned}
$$

将 TD 残差形式转换为优势函数 $A^{\pi_\theta}$：

$$
J(\theta^{\prime})-J(\theta)=\mathbb{E}_{\pi_{\theta^{\prime}}}\left[\sum_{t=0}^\infty\gamma^tA^{\pi_\theta}(s_t,a_t)\right]
$$

进一步展开为期望形式：

$$
J(\theta^{\prime})-J(\theta)=\sum_\tau\left[p(\tau|\pi_{\theta^{\prime}})\sum_{t=0}^\infty\gamma^tA^{\pi_\theta}(s_t,a_t)\right]
$$

- 轨迹概率：$p(\tau|\theta)=p(s_0)\prod_{t=0}^{T}[\pi_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)]$

由于状态访问分布定义为
$\nu^\pi(s)=(1-\gamma)\sum_{t=0}^\infty\gamma^tP(s_t = s, a_t = a \| \pi)$，
因此还可以进一步写成状态访问概率分布的形式：

$$
J(\theta^{\prime})-J(\theta) = \frac{1}{1 - \gamma} \sum_{s} \left[ \nu^{\pi_{\theta'}}(s) \sum_{a} \left[ \pi_{\theta'}(a | s)   A^{\pi_\theta}(s, a)  \right] \right]
$$

因此，只需保证：

$$
\sum_{s} \left[ \nu^{\pi_{\theta'}}(s) \sum_{a} \left[ \pi_{\theta'}(a | s)   A^{\pi_\theta}(s, a)  \right] \right] \geq 0
$$

- 该条件能够保证策略性能单调提升。

然而，为所有可能的新策略采样数据，以获得对应的状态访问分布，
再评估哪些新策略满足上述条件，显然是不现实的。
TRPO 采用近似处理：忽略新旧策略之间状态访问分布的变化，
直接使用旧策略的状态分布 $\nu^{\pi_{\theta}}(s)$：

$$
J(\theta^{\prime})-J(\theta) = \frac{1}{1 - \gamma} \sum_{s} \left[ \nu^{\pi_{\theta}}(s) \sum_{a} \left[ \pi_{\theta'}(a | s)   A^{\pi_\theta}(s, a)  \right] \right]
$$

- 当新旧策略非常接近时，状态访问分布的变化较小，因此这种近似是合理的。

由此可定义如下**优化目标**：

$$
\begin{aligned}
L_\theta(\theta^{\prime})& =J(\theta)+\frac{1}{1 - \gamma} \sum_{s} \left[ \nu^{\pi_{\theta}}(s) \sum_{a} \left[ \pi_{\theta'}(a | s)   A^{\pi_\theta}(s, a)  \right] \right] \\
& =J(\theta)+\frac{1}{1 - \gamma} \sum_{s} \left[ \nu^{\pi_{\theta}}(s) \sum_{a}\pi_{\theta}(a | s) \left[    \frac{\pi_{\theta^{\prime}}(a|s)}{\pi_\theta(a|s)}A^{\pi_\theta}(s,a) \right] \right] \\
& =J(\theta)+\mathbb{E}_{s\sim\nu^{\pi_\theta}}\mathbb{E}_{a\sim\pi_\theta(\cdot|s)}\left[\frac{\pi_{\theta^{\prime}}(a|s)}{\pi_\theta(a|s)}A^{\pi_\theta}(s,a)\right]
\end{aligned}
$$

### 策略更新范围约束

在 TRPO 中，使用 **KL 散度**限制每次策略更新的幅度，
确保新策略与旧策略之间的差异不会过大，
从而维持优化过程的稳定性。
具体而言，TRPO 在每次策略更新时引入如下基于 KL 散度的约束：

$$
\sum_s\nu^{\pi_\theta}(s)\mathrm{KL}\left[\pi_\theta(\cdot|s)||\pi_{\theta^{\prime}}(\cdot|s)\right]=\mathbb{E}_{s\sim\nu^{\pi_\theta}} \left[ \text{KL} \left( \pi_\theta(\cdot | s) || \pi_{\theta'}(\cdot | s) \right) \right] \leq \delta
$$

- $\delta$：新旧策略差异的上界约束。

该不等式约束在策略空间中定义了一个“KL 球”，也就是所谓的**信赖域**。
在该区域内，假设当前学习策略与环境交互产生的状态分布，
与上一轮旧策略采样得到的状态分布基本一致，
从而实现当前策略的稳定改进。

## PPO-KL

TRPO 使用 Taylor 展开近似、共轭梯度和线搜索等方法，直接求解以下约束优化问题：

$$
\begin{aligned}
\max_{\theta} \quad & \mathbb{E}_{s\sim\nu^{\pi_{\theta_k}}}\mathbb{E}_{a\sim\pi_{\theta_k}(\cdot|s)}\left[\frac{\pi_{\theta^{\prime}}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a)\right] \\
\text{subject to} \quad & \mathbb{E}_{s\sim\nu^{\pi_{\theta_k}}} \left[ D_{KL} \left( \pi_{\theta_k}(\cdot|s), \pi_{\theta^{\prime}}(\cdot|s) \right) \right] \leq \delta
\end{aligned}
$$

- 然而，TRPO 的计算复杂度较高，尤其是在涉及 Hessian 矩阵计算和二阶优化时。

相比之下，PPO 使用更简单且高效的方法实现相同目标。
这些方法主要分为两类：**裁剪代理目标（Clipped Surrogate Objective）和自适应 KL 惩罚（Adaptive KL Penalty）**。

下面介绍**自适应 KL 惩罚**。

### 自适应 KL 惩罚

**PPO-KL** 使用拉格朗日乘子法，将 KL 散度约束直接加入目标函数，
从而把原来的约束优化问题转化为无约束优化问题。
在迭代过程中，KL 散度前的系数会被持续更新：

$$
\arg\max_{\theta}\mathbb{E}_{s\sim\nu}\mathbb{E}_{a\sim\pi_{\theta_{k}}(\cdot|s)}\left[\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{k}}(a|s)}A^{\pi_{\theta_{k}}}(s,a)-\beta D_{KL}[\pi_{\theta_{k}}(\cdot|s),\pi_{\theta}(\cdot|s)]\right]
$$

记：

- $d_k = D_{KL}^{\nu_{\theta_k}}(\pi_{\theta_k}, \pi_{\theta})$ 表示当前策略 $\pi_{\theta}$ 与旧策略 $\pi_{\theta_k}$ 之间的平均 KL 散度。
- $\beta_k$ 表示当前迭代中的 KL 惩罚系数。
- $\delta$ 是预先设定的超参数，用于限制当前学习策略与上一轮策略之间的差异。

在 TRPO 中，KL 散度是一个硬约束；而在 PPO-KL 中，它按照如下方式进行自适应调整：

| 当前 KL 距离 $d_k$ 与目标值 $\delta$ 的关系 | 更新后的 $\beta$ |
|---------------------------------------------|-------------------|
| $d_k < \delta / 1.5$ | $\beta_{k+1} = \beta_k / 2$ |
| $d_k > \delta \times 1.5$ | $\beta_{k+1} = \beta_k \times 2$ |
| 其他情况 | $\beta_{k+1} = \beta_k$ |

## 算法

下面给出了使用固定长度轨迹片段的近端策略优化（PPO）算法：

![算法 1](./../../../_static/figures/pseucodes/pseucode-PPO.png)

**算法 1：** 在每次迭代中，$N$ 个并行执行器分别收集 $T$ 个时间步的经验。
随后，基于汇总后的 $N\times T$ 个时间步数据构造代理损失，
并使用小批量随机梯度下降（SGD）优化 $K$ 个轮次。

## 在 XuanCe 中运行 PPO

在 XuanCe 中运行 **PPO** 之前，需要先准备一个 conda 环境，并按照
[**安装步骤**](./../../usage/installation.rst)安装 ``xuance``。

### 运行内置示例

完成安装后，可以打开 Python 控制台，并使用以下命令直接运行 **PPO**：

```python3
import xuance
runner = xuance.get_runner(algo='ppo',  # 注意：默认的 yaml 文件使用 PPO
                           env='classic_control',  # 可选项：classic_control、box2d、atari 等。
                           env_id='CartPole-v1',  # 可选项：CartPole-v1、Pendulum-v1 等。
                          )
runner.run()  # 也可以使用 runner.benchmark()
```

### 使用自定义配置运行

如需使用不同配置运行 **PPO**，例如 PPO、PPO_KL 或其他配置，
可以新建一个 ``.yaml`` 文件，例如 ``my_config.yaml``。
然后使用以下代码运行 **PPO**：

```python3
import xuance
runner = xuance.get_runner(algo='ppo',
                           env='classic_control',  # 可选项：classic_control、box2d、atari 等。
                           env_id='CartPole-v1',  # 可选项：CartPole-v1、Pendulum-v1 等。
                           config_path="my_config.yaml",  # 请确保 my_config.yaml 文件的路径正确。
                          )
runner.run()  # 也可以使用 runner.benchmark()
```

如需进一步了解配置方法，请参阅
[**配置教程**](./../../api/configs/configuration_examples.rst)。

### 在自定义环境中运行

如需在 XuanCe 尚未包含的自定义环境中运行 **PPO**，
需要按照[**新环境教程**](./../../usage/custom_env/custom_drl_env.rst)
中的步骤定义新环境。
然后，[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations)
``ppo_myenv.yaml``。

完成上述操作后，可以使用以下代码在自定义环境中运行 **PPO**：

```python3
import argparse
from xuance.common import load_yaml
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import PPOKL_Agent

configs_dict = load_yaml(file_dir="ppo_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # 创建并行环境。
Agent = PPOKL_Agent(config=configs, envs=envs)  # 创建一个来自 XuanCe 的 PPO 智能体。
Agent.train(configs.running_steps // configs.parallels)  # 对模型进行多个步骤的训练。
Agent.save_model("final_train_model.pth")  # 将模型保存到 model_dir。
Agent.finish()  # 结束训练。
```

## 参考文献

```{code-block} bash
@article{schulman2017proximal,
  title={Proximal policy optimization algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}
```
