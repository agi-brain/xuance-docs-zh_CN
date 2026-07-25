# Double Deep Q-Network (Double DQN)

**论文链接：**  
[**AAAI**](https://ojs.aaai.org/index.php/AAAI/article/view/10295)，
[**ArXiv**](https://arxiv.org/pdf/1509.06461)，
[**Double Q-learning**](https://proceedings.neurips.cc/paper_files/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf)

双重深度 Q 网络（Double Deep Q-Network，Double DQN）是对原始深度 Q 网络（Deep Q-Network，DQN）算法的改进，
旨在解决基于 Q-learning 的方法中常见的 Q 值高估问题。
当 Q-learning 算法根据最大 Q 值选择动作时，由于噪声或近似误差，该最大 Q 值可能被高估，从而产生高估问题。
这可能导致次优策略以及训练过程不稳定。

下表列出了 Double DQN 算法的一些基本特征：

| Double DQN 的特征 | 是否支持 | 说明 |
|------------------|----------|------|
| 同策略（On-policy） | ❌ | 评估策略与目标策略相同。 |
| 异策略（Off-policy） | ✅ | 评估策略与目标策略不同。 |
| 无模型（Model-free） | ✅ | 无需预先构建环境动力学模型。 |
| 基于模型（Model-based） | ❌ | 需要环境模型来训练策略。 |
| 离散动作 | ✅ | 可处理离散动作空间。 |
| 连续动作 | ❌ | 可处理连续动作空间。 |

## 高估的风险

在标准 DQN 中，由于动作选择和动作评估均使用同一个 Q 网络，因此会产生高估问题。

如前文所述，[**DQN**](dqn.md) 使用下一状态的最大 Q 值
$\max_{a'}Q(s', a')$ 作为目标值的一部分，从而更新状态—动作对的 Q 值 $Q(s, a)$。

当 Q 网络高估了一个或多个状态—动作对的价值时，这种高估会随着时间传播并不断累积。
其结果是 Q 值变得过于乐观，不仅可能造成训练不稳定，还会使策略倾向于选择次优动作。

高估会使智能体偏向那些看起来比实际情况更优的动作，
进而可能在复杂环境中导致错误决策和性能下降。

此外，如果高估现象过于严重，还可能破坏训练过程的稳定性，导致 Q 值发散。

## 核心思想

Double DQN 的核心思想是将动作选择与动作评估过程解耦，
从而降低 Q 值被高估的风险。
具体而言，它通过两个相互分离的 Q 值估计步骤实现这一目标：

- **动作选择**：使用当前 Q 网络为给定状态选择最优动作。

$$
a^* = \arg\max_{a}Q(s', s; \theta).
$$

- **动作评估**：使用目标 Q 网络评估所选动作的价值。

$$
y = r + \gamma Q(s', a*;\theta^{-}).
$$

随后，通过最小化以下损失函数来更新 Q 网络：

$$
L(\theta) = \mathbb{E}_{(s, a, s', r) \sim \mathcal{D}}[(y-Q(s, a; \theta))^2].
$$

最后，不要忘记更新目标网络：$\theta^{-} \leftarrow \theta$。

## 与 DQN 的比较

为了更清楚地说明 DQN 与 Double DQN 之间的区别，
可以分别改写二者在 DQN 中的
[目标值计算方式](dqn.md)，
并对比以下公式：

$$
\mathrm{DQN}: y = r + \gamma Q(s', \arg\max_{a'}Q(s', a'; \theta^{-}); \theta^{-})
$$

$$
\mathrm{Double\ DQN}: y = r + \gamma Q(s', \arg\max_{a'}Q(s', a'; \theta); \theta^{-})
$$

可以看出，DQN 与 Double DQN 的主要区别，在于由目标 Q 网络负责评估的贪心动作是如何确定的。
DQN 根据目标 Q 网络本身选择贪心动作，
而 Double DQN 则根据在线 Q 网络选择贪心动作，再由目标 Q 网络对其进行评估。

## 框架

XuanCe 中实现的 Double DQN，其智能体与环境之间的整体交互过程如下图所示。

```{eval-rst}
.. image:: ./../../../_static/figures/algo_framework/dqn_framework.png
    :width: 100%
    :align: center
```

## 在 XuanCe 中运行 Double DQN

在 XuanCe 中运行 Double DQN 之前，需要先准备 conda 环境，并按照
[**安装步骤**](./../../usage/installation.rst)安装 ``xuance``。

### 运行内置示例

完成安装后，可以打开 Python 控制台，并使用以下命令直接运行 Double DQN：

```python3
import xuance
runner = xuance.get_runner(algo='ddqn',
                           env='classic_control',  # 可选项：classic_control、box2d、atari。
                           env_id='CartPole-v1',  # 可选项：CartPole-v1、LunarLander-v3、ALE/Breakout-v5 等。
                           )
runner.run()  # 也可以使用 runner.benchmark()
```

### 使用自定义配置运行

如果希望使用不同的配置运行 Double DQN，可以新建一个 ``.yaml`` 文件，例如 ``my_config.yaml``。
然后，通过以下代码运行 Double DQN：

```python3
import xuance
runner = xuance.get_runner(algo='ddqn',
                       env='classic_control',  # 可选项：classic_control、box2d、atari。
                       env_id='CartPole-v1',  # 可选项：CartPole-v1、LunarLander-v3、ALE/Breakout-v5 等。
                       config_path="my_config.yaml",  # 请确保 my_config.yaml 文件的路径正确。
                       )
runner.run()  # 也可以使用 runner.benchmark()
```

要进一步了解配置方法，请参阅
[**配置教程**](./../../api/configs/configuration_examples.rst)。

### 在自定义环境中运行

如果希望在 XuanCe 尚未内置的自定义环境中运行 Double DQN，
需要按照
[**新环境教程**](./../../usage/custom_env/custom_drl_env.rst)中的步骤定义新环境。
随后，[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations)
``ddqn_myenv.yaml``。

完成上述操作后，可以使用以下代码在自定义环境中运行 Double DQN：

```python3
import argparse
from xuance.common import load_yaml
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import DDQN_Agent

configs_dict = load_yaml(file_dir="dqn_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # 创建并行环境。
Agent = DDQN_Agent(config=configs, envs=envs)  # 创建 XuanCe 的 DDQN 智能体。
Agent.train(configs.running_steps // configs.parallels)  # 训练模型若干步。
Agent.save_model("final_train_model.pth")  # 将模型保存至 model_dir。
Agent.finish()  # 结束训练并释放相关资源。
```

## 引用文献

```{code-block} bash
@article{hasselt2010double,
  title={Double Q-learning},
  author={Hasselt, Hado},
  journal={Advances in neural information processing systems},
  volume={23},
  year={2010}
}
```

```{code-block} bash
@inproceedings{van2016deep,
  title={Deep reinforcement learning with double q-learning},
  author={Van Hasselt, Hado and Guez, Arthur and Silver, David},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={30},
  number={1},
  year={2016}
}
```
