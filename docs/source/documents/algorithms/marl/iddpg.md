# 独立深度确定性策略梯度 (IDDPG)

**论文链接:** [**https://arxiv.org/abs/1706.02275**](https://arxiv.org/abs/1706.02275)

独立深度确定性策略梯度（I-DDPG）是一种多智能体深度强化学习算法，
它以 DDPG 算法为基线，采用完全去中心化作为训练方法。
这意味着将多个单智能体应用于多智能体环境，
每个智能体独立学习，从每个智能体的角度来看，其他智能体的决策是环境的一部分。

下表列出了 IDDPG 算法的一些一般特性：

| IDDPG 的特性                                       | 值   | 描述                                                                                                   |
|----------------------------------------------------|------|-------------------------------------------------------------------------------------------------------|
| 完全去中心化                                        | ✅    | 智能体之间没有通信。                                                                                         |
| 完全中心化                                          | ❌    | 智能体将所有信息发送给中央控制器，控制器将为所有智能体做出决策。                                                          |
| 中心化训练与去中心化执行（CTDE）                          | ❌    | 在训练中使用中央控制器，在执行中放弃。                                                                        |
| 同策略                                               | ❌    | 评估策略与目标策略相同。                                                                                     |
| 异策略                                               | ✅    | 评估策略与目标策略不同。                                                                                     |
| 无模型                                               | ✅    | 不需要准备环境动力学模型。                                                                                   |
| 基于模型                                             | ❌    | 需要环境模型来训练策略。                                                                                     |
| 离散动作                                             | ✅    | 处理离散动作空间。                                                                                         |
| 连续动作                                             | ✅    | 处理连续动作空间。                                                                                         |

## IDDPG 的核心思想

每个智能体独立训练自己的执行器和评判器网络，
**不共享参数**，并且**不直接感知其他智能体的策略**。

### 评判器更新

评判器的目标是最小化 TD 误差，类似于 DDPG：

$$
\mathcal{L}(\phi_i)=\mathbb{E}_{(o_i,a_i,r_i,\sigma_i^{\prime})\sim\mathcal{D}}\left[\left(Q_i(o_i,a_i;\phi_i)-y_i\right)^2\right]
$$

其中 $y_i=r_i+\gamma Q_i^\prime(o_i^\prime,\pi_i^\prime(o_i^\prime;\theta_i^\prime))$，$\phi_i$ 表示 Q 网络参数，$\theta_i^\prime$ 表示策略目标网络参数。

### 执行器更新

执行器的策略梯度方向类似于 DDPG：

$$
\nabla_{\theta_i}J(\theta_i)=\mathbb{E}_{o_i\sim\mathcal{D}}\left[\nabla_{\theta_i}\pi_i(o_i;\theta_i)\nabla_{a_i}Q_i(o_i,a_i;\phi_i)|_{a_i=\pi_i(o_i;\theta_i)}\right]
$$

每个智能体通过其评判器的梯度反馈独立更新执行器。

## 在 XuanCe 中运行 IDDPG

在 XuanCe 中运行 IDDPG 之前，您需要准备 conda 环境并按照 
[**安装步骤**](./../../usage/installation.rst#install-xuance) 安装 ``xuance``。

### 运行内置演示

完成安装后，您可以打开 Python 控制台并使用以下命令直接运行 IDDPG：

```python3
import xuance
runner = xuance.get_runner(method='iddpg',
                           env='mpe',
                           env_id='simple_spread_v3',
                           is_test=False)
runner.run()  # 或 runner.benchmark()
```

### 使用自定义配置运行

如果您想使用不同的配置运行 IDDPG，可以创建一个新的 ``.yaml`` 文件，例如 ``my_config.yaml``。
然后，通过以下代码块运行 IDDPG：

```python3
import xuance
runner = xuance.get_runner(method='iddpg',
                       env='mpe',
                       env_id='simple_spread_v3',
                       config_path="my_config.yaml",
                       is_test=False)
runner.run()  # 或 runner.benchmark()
```

要了解更多关于配置的信息，请访问 
[**配置教程**](./../../api/configs/configuration_examples.rst)。


### 使用自定义环境运行

如果您想在 XuanCe 中未包含的自己的环境中运行 XuanCe 的 IDDPG， 
您需要按照
[**新环境教程**](./../../usage/custom_env/custom_drl_env.rst) 中的步骤定义新环境。
然后，[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``iddpg_myenv.yaml``。

之后，您可以使用以下代码在自己的环境中运行 IDDPG：

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import IDDPG_Agents

configs_dict = get_configs(file_dir="iddpg_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # 创建并行环境。
Agent = IDDPG_Agents(config=configs, envs=envs)  # 从 XuanCe 创建 IDDPG 智能体。
Agent.train(configs.running_steps // configs.parallels)  # 训练模型多个步骤。
Agent.save_model("final_train_model.pth")  # 将模型保存到 model_dir。
Agent.finish()  # 完成训练。
```

## 引用

```{code-block} bash
@article{lowe2017multi,
  title={Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
  author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
  journal={Neural Information Processing Systems (NIPS)},
  year={2017}
}
```

