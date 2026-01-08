# 独立软执行器-评判器 (ISAC)

**论文链接:** [**https://arxiv.org/abs/2104.06655**](https://arxiv.org/abs/2104.06655)

独立软执行器-评判器（I-SAC）是一种多智能体深度强化学习算法，
它以 SAC 算法为基线，采用完全去中心化作为训练方法。
去中心化决策避免了动作空间的指数增长，从而提高了学习效率。

下表列出了 ISAC 算法的一些一般特性：

| ISAC 的特性                                       | 值   | 描述                                                                                                   |
|---------------------------------------------------|------|-------------------------------------------------------------------------------------------------------|
| 完全去中心化                                        | ✅    | 智能体之间没有通信。                                                                                         |
| 完全中心化                                          | ❌    | 智能体将所有信息发送给中央控制器，控制器将为所有智能体做出决策。                                                          |
| 中心化训练与去中心化执行（CTDE）                          | ❌    | 在训练中使用中央控制器，在执行中放弃。                                                                        |
| 同策略                                               | ❌    | 评估策略与目标策略相同。                                                                                     |
| 异策略                                               | ✅    | 评估策略与目标策略不同。                                                                                     |
| 无模型                                               | ✅    | 不需要准备环境动力学模型。                                                                                   |
| 基于模型                                             | ❌    | 需要环境模型来训练策略。                                                                                     |
| 离散动作                                             | ✅    | 处理离散动作空间。                                                                                         |
| 连续动作                                             | ✅    | 处理连续动作空间。                                                                                         |

## ISAC 的核心思想

### 评判器更新

评判器的目标是最小化 TD 误差，类似于 SAC：

$$
\mathcal{L}_{Q^i}(\phi)=\mathbb{E}_{(o_t,a_t)\sim D}[\frac{1}{2}(Q^i(a_t,o_t)-y_t)^2],\forall i\in\{1,2\}
$$

其中 $y_t=r_t+(1-d_t)\gamma\mathbb{E}_{a_{t+1}\sim\pi}[\min_{j\in\{1,2\}}(\hat{Q}^j(a_{t+1},o_{t+1}))-\alpha\log\pi(a_{t+1}|o_{t+1})]$，
$Q^i(a_t,o_t)$ 表示 Q 网络的值。

### 执行器更新

策略通过最小化一个结合期望回报和策略熵的损失函数进行优化，鼓励高奖励和充分探索。
执行器损失函数表述为：

$$
\mathcal{J}_\pi(\theta_i)=\mathbb{E}_{o_t\sim D,a_t\sim\pi}[\alpha\log\pi_{\theta_i}(a_t|o_t)-\min_{j\in\{1,2\}}(Q^j(a_t,o_t))]
$$

这与 SAC 类似。

## 算法

ISAC 的完整训练算法如算法 1 所示：

**算法 1：ISAC 算法**  
**输入：**
共享回放缓冲区 $D$（容量 $N$）；
策略网络函数 $\pi$（参数 $\theta$）；
评判器网络函数 $Q^1$，$Q^2$（参数 $\phi_1$，$\phi_2$）；
目标评判器网络函数 $\hat{Q}^1$，$\hat{Q}^2$，初始化为 $\hat{\phi}_i = \phi_i$。  
**输出：**
训练后的策略 $\pi$。

**For** episode = 0,...,M **Do**  
&nbsp;&nbsp;&nbsp;&nbsp;t=0;初始化状态 $o_t$  
&nbsp;&nbsp;&nbsp;&nbsp;**While** t < TimeLimit **Do**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**For** agent = 0,...,K **Do**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;采样动作 $a_k^t \sim \pi(o_k^t)$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**End For**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;将动作 $a_k^t$ 组合成联合动作 $a_t$    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;执行 $a_t$，观察奖励 $r_t$，下一个观测 $o_{t+1}$ 和完成标志 $d_{t+1}$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;将 $(o_t, a_t, r_t, o_{t+1}, d_{t+1})$ 存储在缓冲区 $D$ 中  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**For** e = 0,...,E **Do**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;从 $D$ 中采样批次  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;计算目标 $y_t=r_t+(1-d_t)\gamma\mathbb{E}_{a_{t+1}\sim\pi}[\min_{j\in\{1,2\}}(\hat{Q}^j(a_{t+1},o_{t+1}))-\alpha\log\pi(a_{t+1}|o_{t+1})]$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;更新 $Q^i$ 的权重 $\phi_i\leftarrow\phi_i-\omega\nabla \mathcal{L}_{q_i}(\phi),\quad\forall i\in\{1,2\}$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;更新 $\pi$ 的权重 $\theta\leftarrow\theta-\lambda\nabla \mathcal{J}_{\pi}(\theta)$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;更新 $\hat{Q}^i$ 的权重 $\hat{\phi}_i \leftarrow \tau \phi_i + (1-\tau)\hat{\phi}_i, \quad\forall i\in\{1,2\}$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**End For**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**If** $d_{t+1}$ **then**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;break  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**End If**  
&nbsp;&nbsp;&nbsp;&nbsp;**End While**  
**End For**

## 在 XuanCe 中运行 ISAC

在 XuanCe 中运行 ISAC 之前，您需要准备 conda 环境并按照 
[**安装步骤**](./../../usage/installation.rst#install-xuance) 安装 ``xuance``。

### 运行内置演示

完成安装后，您可以打开 Python 控制台并使用以下命令直接运行 ISAC：

```python3
import xuance
runner = xuance.get_runner(method='isac',
                           env='mpe',
                           env_id='simple_spread_v3',
                           is_test=False)
runner.run()  # 或 runner.benchmark()
```

### 使用自定义配置运行

如果您想使用不同的配置运行 ISAC，可以创建一个新的 ``.yaml`` 文件，例如 ``my_config.yaml``。
然后，通过以下代码块运行 ISAC：

```python3
import xuance
runner = xuance.get_runner(method='isac',
                       env='mpe',
                       env_id='simple_spread_v3',
                       config_path="my_config.yaml",
                       is_test=False)
runner.run()  # 或 runner.benchmark()
```

要了解更多关于配置的信息，请访问 
[**配置教程**](./../../api/configs/configuration_examples.rst)。


### 使用自定义环境运行

如果您想在 XuanCe 中未包含的自己的环境中运行 XuanCe 的 ISAC， 
您需要按照
[**新环境教程**](./../../usage/custom_env/custom_drl_env.rst) 中的步骤定义新环境。
然后，[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``isac_myenv.yaml``。

之后，您可以使用以下代码在自己的环境中运行 ISAC：

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import ISAC_Agents

configs_dict = get_configs(file_dir="isac_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # 创建并行环境。
Agent = ISAC_Agents(config=configs, envs=envs)  # 从 XuanCe 创建 ISAC 智能体。
Agent.train(configs.running_steps // configs.parallels)  # 训练模型多个步骤。
Agent.save_model("final_train_model.pth")  # 将模型保存到 model_dir。
Agent.finish()  # 完成训练。
```

## 引用

```{code-block} bash
@misc{pu2021decomposedsoftactorcriticmethod,
      title={Decomposed Soft Actor-Critic Method for Cooperative Multi-Agent Reinforcement Learning}, 
      author={Yuan Pu and Shaochen Wang and Rui Yang and Xin Yao and Bin Li},
      year={2021},
      eprint={2104.06655},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2104.06655}, 
}
```

