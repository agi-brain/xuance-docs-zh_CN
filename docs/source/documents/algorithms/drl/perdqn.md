# DQN with Prioritized Experience Replay (PerDQN)

**论文链接：** [**https://arxiv.org/pdf/1511.05952**](https://arxiv.org/pdf/1511.05952)

基于优先经验回放的 DQN（PER DQN）是传统 DQN 的一种变体。
该算法引入优先经验回放机制，通过在训练过程中优先使用某些经验，
提升智能体的学习效率。

下表列出了 PER DQN 算法的一些基本特征：

| PER DQN 的特征       | 是否具备 | 说明               |
|-------------------|------|------------------|
| 同策略（On-policy）    | ❌    | 评估策略与目标策略相同。     |
| 异策略（Off-policy）   | ✅    | 评估策略与目标策略不同。     |
| 无模型（Model-free）   | ✅    | 无须预先构建环境动力学模型。   |
| 基于模型（Model-based） | ❌    | 需要使用环境模型训练策略。    |
| 离散动作              | ✅    | 可处理离散动作空间。       |
| 连续动作              | ❌    | 可处理连续动作空间。       |

## 方法

在标准的 [**DQN**](./dqn.md#deep-q-netowrk) 中，经验被存储在经验回放缓冲区中，
智能体从缓冲区中均匀采样经验，以训练其 Q 网络。
然而，均匀采样的效率可能较低，尤其是当某些经验对学习比其他经验更加重要时。
PER DQN 通过优先采样那些预计能够为策略改进提供更多有效信息的经验，解决这一问题。

### 优先经验回放

- 在优先经验回放（Prioritized Experience Replay，PER）中，经验不再从缓冲区中被均匀采样，而是根据其时序差分（Temporal-Difference，TD）误差确定优先级。
- TD 误差是期望 Q 值（由 Bellman 方程得到）与智能体 Q 网络当前预测的 Q 值之间的差异。
- 较大的 TD 误差意味着该经验具有较高的学习潜力，因为它表明智能体当前的 Q 函数未能准确预测该经验对应的未来回报。

### PER DQN 的工作原理

- 在 PER DQN 中，经验回放缓冲区增加了优先采样机制。某条经验的优先级与其 TD 误差成正比。
- 当智能体为训练采样经验时，TD 误差较大的经验更有可能被选中。
- 这种方式使智能体将学习重点放在更出乎预期或更困难的经验上，并通过更加频繁地重新使用重要经验来加快学习过程。

### 重要性采样

为了避免因优先采样经验而使训练过程产生偏差，PER DQN 使用了重要性采样。
每条经验都会被赋予一个用于补偿非均匀采样的权重。
这可以确保即使经验不是均匀采样的，智能体仍然能够进行正确学习。

### 数学细节

经验 $i$ 的优先级 $p_i$ 使用 TD 误差 $\delta_i$ 计算，通常表示为：

$$
p_i = |\delta_i| + \epsilon,
$$

其中，$|\delta_i|$ 是 TD 误差的绝对值；
$\epsilon$ 是一个较小的常数，用于确保 TD 误差为零的经验仍然能够被保留在缓冲区中并参与采样。

经验 $i$ 被采样的概率为：

$$
P(i) = \frac{p^{\alpha}_i}{\sum_{k}{p^{\alpha}_{k}}},
$$

其中，$\alpha$ 用于控制优先采样的程度，即 TD 误差对采样概率的影响程度。

## 算法

用于训练 PER DQN 的完整算法如算法 1 所示：

```{eval-rst}
.. image:: ./../../../_static/figures/pseucodes/pseucode-PERDQN.png
    :width: 88%
    :align: center
```

## 在 XuanCe 中运行 PER DQN

在 XuanCe 中运行 PER DQN 之前，需要先准备一个 conda 环境，并按照
[**安装步骤**](./../../usage/installation.rst)安装 ``xuance``。

### 运行内置示例

完成安装后，可以打开 Python 控制台，并使用以下命令直接运行 PER DQN：

```python3
import xuance
runner = xuance.get_runner(method='perdqn',
                           env='classic_control',  # 可选项：claasi_control、box2d、atari。
                           env_id='CartPole-v1',  # 可选项：CartPole-v1、LunarLander-v2、ALE/Breakout-v5 等。
                           is_test=False)
runner.run()  # 也可以使用 runner.benchmark()
```

### 使用自定义配置运行

如需使用不同配置运行 PER DQN，可以新建一个 ``.yaml`` 文件，例如 ``my_config.yaml``。
然后使用以下代码运行 PER DQN：

```python3
import xuance as xp
runner = xp.get_runner(method='perdqn',
                       env='classic_control',  # 可选项：claasi_control、box2d、atari。
                       env_id='CartPole-v1',  # 可选项：CartPole-v1、LunarLander-v2、ALE/Breakout-v5 等。
                       config_path="my_config.yaml",  # 请确保 my_config.yaml 文件的路径正确。
                       is_test=False)
runner.run()  # 也可以使用 runner.benchmark()
```

如需进一步了解配置方法，请参阅
[**配置教程**](./../../api/configs/configuration_examples.rst)。

### 在自定义环境中运行

如需在 XuanCe 尚未包含的自定义环境中运行 PER DQN，
需要按照 [**新环境教程**](./../../usage/custom_env/custom_drl_env.rst)
中的步骤定义新环境。
然后，[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations)
``perdqn_myenv.yaml``。

完成上述操作后，可以使用以下代码在自定义环境中运行 PER DQN：

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import PerDQN_Agent

configs_dict = get_configs(file_dir="perdqn_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # 创建并行环境。
Agent = PerDQN_Agent(config=configs, envs=envs)  # 创建一个来自 XuanCe 的 DDPG 智能体。
Agent.train(configs.running_steps // configs.parallels)  # 对模型进行多个步骤的训练。
Agent.save_model("final_train_model.pth")  # 将模型保存到 model_dir。
Agent.finish()  # 结束训练。
```

## 参考文献

```{code-block} bash
@inproceedings{DBLP:journals/corr/SchaulQAS15,
  author       = {Tom Schaul and
                  John Quan and
                  Ioannis Antonoglou and
                  David Silver},
  editor       = {Yoshua Bengio and
                  Yann LeCun},
  title        = {Prioritized Experience Replay},
  booktitle    = {4th International Conference on Learning Representations, {ICLR} 2016,
                  San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings},
  year         = {2016},
  url          = {http://arxiv.org/abs/1511.05952},
  timestamp    = {Thu, 25 Jul 2019 14:25:38 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/SchaulQAS15.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
