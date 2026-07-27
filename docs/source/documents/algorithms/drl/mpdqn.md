# 多通道参数化深度 Q 网络（MP-DQN）

**算法全称**：**Multi-pass Parameterised Deep Q-Network (MP-DQN)**

**论文链接：** [**https://arxiv.org/abs/1905.04388**](https://arxiv.org/abs/1905.04388)

多通道参数化深度 Q 网络（Multi-pass Parameterised Deep Q-Network，MP-DQN）是 DQN 的一种扩展，旨在同时处理离散动作和连续动作参数。该算法面向参数化动作空间问题，其中每个动作由一个离散动作以及与其关联的连续参数共同组成。

下表列出了 MP-DQN 算法的一些基本特征：

| MP-DQN 的特征        | 是否具备 | 说明             |
|-------------------|------|----------------|
| 同策略（On-policy）    | ❌    | 评估策略与目标策略相同。   |
| 异策略（Off-policy）   | ✅    | 评估策略与目标策略不同。   |
| 无模型（Model-free）   | ✅    | 无须预先构建环境动力学模型。 |
| 基于模型（Model-based） | ❌    | 需要使用环境模型训练策略。  |
| 离散动作              | ✅    | 可处理离散动作部分。     |
| 连续动作              | ✅    | 可处理连续动作参数。     |

## 参数化动作空间

参数化动作空间将离散动作与连续参数结合起来。每个离散动作 $k\in K$ 都对应一组连续参数 $x_k\in X_k$。完整动作空间定义为：

$$
\mathcal{A}
=
\bigcup_{k\in[K]}
\left\{
a_k=(k,x_k)\mid x_k\in\mathcal{X}_k
\right\}.
$$

这种混合结构使传统强化学习算法难以直接应用，因为传统算法通常只能处理离散动作或连续动作，而不能同时处理二者。

## MP-DQN 的网络结构

MP-DQN 采用一种新的神经网络结构，主要包括以下部分：

- **共享特征提取器：** 使用公共主干网络处理状态输入。
- **离散动作价值头：** 为每个离散动作估计对应的 Q 值。
- **连续参数网络：** 为每个离散动作 $k$ 预测相应的最优连续参数 $\mu_k(s)$。

网络同时输出离散动作的 Q 值以及相应的连续参数。对应的 Bellman 方程为：

$$
Q(s,k,x_k)
=
\mathbb{E}_{r,s'}
\left[
r+
\gamma
\max_{k'}
Q\left(
s',
k',
x_{k'}^Q(s')
\right)
\mid
s,k,x_k
\right].
$$

## 多通道 Q 值估计

MP-DQN 的核心创新在于使用多通道方式估计 Q 值。

### 前向传播

对于每个离散动作 $k$，首先计算对应的连续参数：

$$
\mu_k(s).
$$

### Q 值计算

针对每个离散动作，分别估计：

$$
Q(s,k,\mu_k(s)).
$$

### 动作选择

选择 Q 值最大的离散动作：

$$
k^*
=
\arg\max_{k\in K}
Q(s,k,\mu_k(s)).
$$

最终执行的参数化动作是：

$$
\left(k^*,\mu_{k^*}(s)\right).
$$

与 P-DQN 将所有动作参数同时输入 Q 网络不同，MP-DQN 针对每个离散动作分别构造一次 Q 网络输入。在计算动作 $k$ 的 Q 值时，仅保留动作 $k$ 对应的连续参数，而将其他动作的参数置零。这样可以减少不同动作参数之间不必要的耦合和干扰。

## 损失函数

MP-DQN 使用两个相互独立的损失函数。

### Q 值损失

Q 网络通过最小时序差分误差进行训练：

$$
L_Q(\theta_Q)
=
\mathbb{E}_{(s,k,x_k,r,s')\sim D}
\left[
\frac{1}{2}
\left(
y-Q(s,k,x_k;\theta_Q)
\right)^2
\right],
$$

其中，目标值 $y$ 为：

$$
y
=
r+
\gamma
\operatorname*{max}_{k'\in[K]}
Q\left(
s',
k',
x_{k'}(s';\theta_x);
\theta_Q
\right).
$$

### 参数策略损失

连续动作参数网络通过最大化各个离散动作对应的 Q 值进行训练：

$$
L_x(\theta_x)
=
-\sum_{k=1}^{K}
Q\left(
s,
k,
\mathbf{x}(s;\theta_x);
\theta_Q
\right).
$$

其中，$\theta_x$ 表示连续动作参数网络的参数。

## 经验回放与目标网络

与 DQN 类似，MP-DQN 采用以下机制：

- **经验回放：** 将转移样本 $(s,k,x_k,r,s')$ 存储在经验回放缓冲区中。
- **目标网络：** 为 Q 网络和连续参数网络分别维护目标网络，并周期性或通过软更新方式更新目标网络参数。
- **$\epsilon$-贪心探索：** 使用 $\epsilon$-贪心策略选择离散动作，并在探索阶段为连续参数采样随机值。

## 算法框架

XuanCe 中实现的 MP-DQN 智能体与环境交互框架如下图所示：

```{eval-rst}
.. image:: ./../../../_static/figures/pseucodes/pseucode-MPDQN.png
    :width: 80%
    :align: center
```

## 在 XuanCe 中运行 MP-DQN

在 XuanCe 中运行 MP-DQN 之前，需要先准备一个 conda 环境，并按照安装说明安装 ``xuance``。

### 运行内置示例

完成安装后，可以打开 Python 控制台，并使用以下命令直接运行 MP-DQN：

```python3
import xuance

runner = xuance.get_runner(
    algo='mpdqn',
    env='parameterised_action_space',  # 可选项：parameterised_action_space。
    env_id='Platform-v0',  # 可选项：Platform-v0、Goal-v0 等。
)
runner.run()  # 也可以使用 runner.benchmark()
```

### 使用自定义配置运行

如需使用不同配置运行 MP-DQN，可以新建一个 ``.yaml`` 文件，例如
``my_mpdqn_config.yaml``。然后使用以下代码运行 MP-DQN：

```python3
import xuance as xp

runner = xp.get_runner(
    algo='mpdqn',
    env='parameterised_action_space',  # 可选项：parameterised_action_space。
    env_id='Platform-v0',  # 可选项：Platform-v0、Goal-v0 等。
    config_path="my_mpdqn_config.yaml",  # 请确保配置文件路径正确。
)
runner.run()  # 也可以使用 runner.benchmark()
```

### 在自定义环境中运行

如需在自定义参数化动作环境中运行 XuanCe 的 MP-DQN，需要按照
[**新环境教程**](./../../usage/custom_env/custom_drl_env.rst)
中的步骤定义新环境。然后，
[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations)
``mpdqn_myenv.yaml``。

完成上述操作后，可以使用以下代码运行 MP-DQN：

```python3
import argparse
from xuance.common import load_yaml
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import MP_DQN_Agent

configs_dict = load_yaml(file_dir="mpdqn_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyParameterisedEnv

envs = make_envs(configs)  # 创建并行环境。
Agent = MP_DQN_Agent(config=configs, envs=envs)  # 创建 MP-DQN 智能体。
Agent.train(configs.running_steps // configs.parallels)  # 训练模型。
Agent.save_model("final_train_model.pth")  # 将模型保存到 model_dir。
Agent.finish()  # 结束训练。
```

## MP-DQN 的优点

- 能够同时处理离散动作和连续动作参数。
- 适用于包含混合动作空间的实际应用。
- 通过经验回放机制提高样本利用效率。
- 通过目标网络提高训练稳定性。
- 将 DQN 的适用范围扩展到参数化动作空间问题。

## 参考文献

```{code-block}
@misc{bester2019multipassqnetworksdeepreinforcement,
      title={Multi-Pass Q-Networks for Deep Reinforcement Learning with Parameterised Action Spaces},
      author={Craig J. Bester and Steven D. James and George D. Konidaris},
      year={2019},
      eprint={1905.04388},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1905.04388},
}
```
