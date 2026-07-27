# 参数化深度 Q 网络（P-DQN）

**算法全称**：**Parameterized Deep Q-Network (P-DQN)**

**论文链接：** [**https://arxiv.org/pdf/1810.06394**](https://arxiv.org/pdf/1810.06394)

参数化深度 Q 网络（Parameterized Deep Q-Network，P-DQN）是一种面向混合动作空间的强化学习框架。该方法无须对混合动作空间进行近似或松弛，而是将用于处理离散动作空间的 DQN 与用于处理连续动作空间的 DDPG 有机结合起来。

下表列出了 P-DQN 算法的一些基本特征：

| P-DQN 的特征         | 是否具备 | 说明              |
|-------------------|------|-----------------|
| 同策略（On-policy）    | ❌    | 评估策略与目标策略相同。    |
| 异策略（Off-policy）   | ✅    | 评估策略与目标策略不同。    |
| 无模型（Model-free）   | ✅    | 无须预先构建环境动力学模型。  |
| 基于模型（Model-based） | ❌    | 需要使用环境模型训练策略。   |
| 离散动作              | ✅    | 可处理离散动作空间。      |
| 连续动作              | ✅    | 可处理连续动作空间。      |
| 混合动作              | ✅    | 可处理离散—连续混合动作空间。 |

## 离散—连续混合动作空间

混合动作可由如下分层结构定义。首先，从离散集合 $[K]$ 中选择一个高层动作 $k$；选定 $k$ 后，再选择与第 $k$ 个高层动作关联的低层连续参数 $x_k\in\mathcal{X}_k$。对于任意 $k\in[K]$，$\mathcal{X}_k$ 均为连续集合。

$$
\mathcal{A}=\{(k,x_k)\mid x_k\in\mathcal{X}_k,\ \forall k\in[K]\}
$$

## P-DQN 的核心思想

在混合动作空间中，将动作价值函数表示为 $Q(s,a)=Q(s,k,x_k)$，其中 $s\in S$、$k\in[K]$，且 $x_k\in\mathcal{X}_k$。设 $k_t$ 为时刻 $t$ 选择的离散动作，$x_{k_t}$ 为与其关联的连续参数，则 Bellman 方程为：

$$
Q(s_t,k_t,x_{k_t})
=
\underset{r_t,s_{t+1}}{\mathbb{E}}
\left[
r_t+
\gamma
\underset{k\in[K]}{\max}
\underset{x_k\in\mathcal{X}_k}{\sup}
Q(s_{t+1},k,x_k)
\mid
s_t=s,\ a_t=(k_t,x_{k_t})
\right].
$$

当函数 $Q$ 固定时，对于任意 $s\in S$ 和 $k\in[K]$，可以将

$$
\underset{x_k\in\mathcal{X}_k}{\operatorname{argsup}}Q(s,k,x_k)
$$

视为一个函数 $x_k^Q:S\rightarrow\mathcal{X}_k$。因此，Bellman 方程可以改写为：

$$
Q(s_t,k_t,x_{k_t})
=
\underset{r_t,s_{t+1}}{\mathbb{E}}
\left[
r_t+
\gamma
\underset{k\in[K]}{\max}
Q(s_{t+1},k,x_k^Q(s_{t+1}))
\mid
s_t=s
\right].
$$

与深度 Q 网络类似，P-DQN 使用深度神经网络 $Q(s,k,x_k;\omega)$ 近似 $Q(s,k,x_k)$，其中 $\omega$ 表示网络参数。对于该动作价值网络，使用确定性策略网络

$$
x_k(\cdot;\theta):S\rightarrow\mathcal{X}_k
$$

近似 $x_k^Q$，其中 $\theta$ 表示策略网络参数。当 $\omega$ 固定时，希望找到参数 $\theta$，使得：

$$
Q(s,k,x_k(s;\theta);\omega)
\approx
\underset{x_k\in\mathcal{X}_k}{\sup}
Q(s,k,x_k;\omega),
\qquad \forall k\in[K].
$$

随后，与 DQN 类似，可以通过梯度下降最小化均方 Bellman 误差，从而估计参数 $\omega$。其 $n$ 步目标值可写为：

$$
y_t=
\sum_{i=0}^{n-1}\gamma^i r_{t+i}
+
\gamma^n
\underset{k\in[K]}{\max}
Q(s_{t+n},k,x_k(s_{t+n};\theta);\omega).
$$

对于动作价值网络参数 $\omega$，使用与 DQN 类似的最小二乘损失。由于在固定 $\omega$ 时，需要寻找使 $Q(s,k,x_k(s;\theta);\omega)$ 最大化的 $\theta$，因此两个损失函数分别为：

$$
\ell^Q(\omega)
=
\frac{1}{2}
\left[
Q(s_t,k_t,x_{k_t};\omega)-y_t
\right]^2,
$$

以及

$$
\ell^\Theta(\theta)
=
-\sum_{k=1}^{K}
Q(s_t,k,x_k(s_t;\theta);\omega_t).
$$

随后使用随机梯度方法更新网络参数。当 $\omega_t$ 固定时，通过最小化损失函数 $\ell^\Theta(\theta)$ 更新连续动作参数网络。

## 算法

训练 P-DQN 的完整算法如算法 1 所示：

```{eval-rst}
.. image:: ./../../../_static/figures/pseucodes/pseucode-PDQN.png
    :width: 100%
    :align: center
```

## 在 XuanCe 中运行 P-DQN

在 XuanCe 中运行 P-DQN 之前，需要先准备一个 conda 环境，并按照
[**安装步骤**](./../../usage/installation.rst)安装 ``xuance``。

### 使用自定义环境运行

如需在 XuanCe 尚未包含的自定义环境中运行 P-DQN，需要按照
[**新环境教程**](./../../usage/custom_env/custom_drl_env.rst)
中的步骤定义新环境。然后，
[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations)
``pdqn_myenv.yaml``。

完成上述操作后，可以使用以下代码在自定义环境中运行 P-DQN：

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import PDQN_Agent

configs_dict = get_configs(file_dir="pdqn_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # 创建并行环境。
Agent = PDQN_Agent(config=configs, envs=envs)  # 创建一个来自 XuanCe 的 P-DQN 智能体。
Agent.train(configs.running_steps // configs.parallels)  # 对模型进行多个步骤的训练。
Agent.save_model("final_train_model.pth")  # 将模型保存到 model_dir。
Agent.finish()  # 结束训练。
```

## 参考文献

```{code-block}
@article{xiong2018parametrized,
  title={Parametrized deep q-networks learning: Reinforcement learning with discrete-continuous hybrid action space},
  author={Xiong, Jiechao and Wang, Qing and Yang, Zhuoran and Sun, Peng and Han, Lei and Zheng, Yang and Fu, Haobo and Zhang, Tong and Liu, Ji and Liu, Han},
  journal={arXiv preprint arXiv:1810.06394},
  year={2018}
}
```
