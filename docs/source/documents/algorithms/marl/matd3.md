# 多智能体双延迟深度确定性策略梯度 (MATD3)

**论文链接:** [**https://arxiv.org/abs/1910.01465**](https://arxiv.org/abs/1910.01465)

多智能体双延迟深度确定性策略梯度（MATD3）将 TD3 扩展到多智能体领域，
类似于将 DDPG 扩展到 MADDPG 的方式。
其核心目标是减少多智能体强化学习中中心化价值函数的高估偏差。

下表列出了 MATD3 算法的一些一般特性：

| MATD3 的特性                                       | 值   | 描述                                                                                                   |
|----------------------------------------------------|------|-------------------------------------------------------------------------------------------------------|
| 完全去中心化                                        | ❌    | 智能体之间没有通信。                                                                                         |
| 完全中心化                                          | ❌    | 智能体将所有信息发送给中央控制器，控制器将为所有智能体做出决策。                                                          |
| 中心化训练与去中心化执行（CTDE）                          | ✅    | 在训练中使用中央控制器，在执行中放弃。                                                                        |
| 同策略                                               | ❌    | 评估策略与目标策略相同。                                                                                     |
| 异策略                                               | ✅    | 评估策略与目标策略不同。                                                                                     |
| 无模型                                               | ✅    | 不需要准备环境动力学模型。                                                                                   |
| 基于模型                                             | ❌    | 需要环境模型来训练策略。                                                                                     |
| 离散动作                                             | ❌    | 处理离散动作空间。                                                                                         |
| 连续动作                                             | ✅    | 处理连续动作空间。                                                                                         |

## MATD3 的核心思想

### 目标策略平滑

在评判器更新中，为所有智能体的动作添加裁剪的高斯噪声 $\epsilon=\mathrm{clip}(\mathcal{N}(0,\sigma),-c,c)$：
$a_{j}^{\prime}=\mu_{\theta_{j}^{\prime}}(o_{j}^{\prime})+\epsilon$。最后，评判器的目标函数表示为：

$$
y_i=r_i+\gamma\min_{j=1,2}Q_{i,\theta_j^{\prime}}^\pi(\mathbf{x}^{\prime},\mu_1^{\prime}(o_1^{\prime})+\epsilon,...,\mu_N^{\prime}(o_N^{\prime})+\epsilon)
$$

其中 $\mu_j^{\prime}$ 是 $\mu_{\theta_{j}^{\prime}}$ 的简写。

### 评判器更新

评判器损失函数可以表示为：

$$
L(\theta_i)=\mathbb{E}_{x,a,r,x^{\prime}}\left[\left(Q_{i,\theta_j}^{\pi} \left(x,a_1,\ldots,a_N\right)-y_i\right)^2\right]
$$

其中 $i$ 表示每个智能体。

### 执行器更新

智能体 $i$ 的确定性策略可以通过梯度下降进行优化：

$$
\nabla_{\theta_i}J\left(\mu_i\right)=\mathbb{E}_{x,a\sim D}\left[\nabla_{\theta_i}\mu_i\left(a_{i}\left|o_{i}\right)\nabla_{a_i}Q_{i,\theta_j}^\mu\left(x,a_1,\ldots,a_N\left|\right._{a_i=\mu_i(o_i)}\right)\right]\right.
$$

其中 $\mu_i$ 是 $\mu_{\theta_{i}}$ 的简写。

## 算法

MATD3 的完整训练算法如算法 1 所示：

```{eval-rst}
.. image:: ./../../../_static/figures/pseucodes/pseucode-MATD3.png
    :width: 80%
    :align: center
```

## 在 XuanCe 中运行 MATD3

在 XuanCe 中运行 MATD3 之前，您需要准备 conda 环境并按照 
[**安装步骤**](./../../usage/installation.rst#install-xuance) 安装 ``xuance``。

### 运行内置演示

完成安装后，您可以打开 Python 控制台并使用以下命令直接运行 MATD3：

```python3
import xuance
runner = xuance.get_runner(method='matd3',
                           env='mpe',
                           env_id='simple_spread_v3',
                           is_test=False)
runner.run()  # 或 runner.benchmark()
```

### 使用自定义配置运行

如果您想使用不同的配置运行 MATD3，可以创建一个新的 ``.yaml`` 文件，例如 ``my_config.yaml``。
然后，通过以下代码块运行 MATD3：

```python3
import xuance
runner = xuance.get_runner(method='matd3',
                       env='mpe',
                       env_id='simple_spread_v3',
                       config_path="my_config.yaml",
                       is_test=False)
runner.run()  # 或 runner.benchmark()
```

要了解更多关于配置的信息，请访问 
[**配置教程**](./../../api/configs/configuration_examples.rst)。


### 使用自定义环境运行

如果您想在 XuanCe 中未包含的自己的环境中运行 XuanCe 的 MATD3， 
您需要按照
[**新环境教程**](./../../usage/custom_env/custom_drl_env.rst) 中的步骤定义新环境。
然后，[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``matd3_myenv.yaml``。

之后，您可以使用以下代码在自己的环境中运行 MATD3：

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import MATD3_Agents

configs_dict = get_configs(file_dir="matd3_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # 创建并行环境。
Agent = MATD3_Agents(config=configs, envs=envs)  # 从 XuanCe 创建 MATD3 智能体。
Agent.train(configs.running_steps // configs.parallels)  # 训练模型多个步骤。
Agent.save_model("final_train_model.pth")  # 将模型保存到 model_dir。
Agent.finish()  # 完成训练。
```


## 引用

```{code-block} bash
@misc{ackermann2019reducingoverestimationbiasmultiagent,
      title={Reducing Overestimation Bias in Multi-Agent Domains Using Double Centralized Critics}, 
      author={Johannes Ackermann and Volker Gabler and Takayuki Osa and Masashi Sugiyama},
      year={2019},
      eprint={1910.01465},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1910.01465}, 
}

```

