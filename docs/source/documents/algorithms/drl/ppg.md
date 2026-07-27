# 分阶段策略梯度（PPG）

**论文链接：** [**https://proceedings.mlr.press/v139/cobbe21a**](https://proceedings.mlr.press/v139/cobbe21a)

分阶段策略梯度（Phasic Policy Gradient，PPG）是一种先进的强化学习算法，旨在提高策略优化的效率。
该算法建立在
[**PPO**](ppoclip.md)
框架之上，通过引入两阶段训练机制，
将策略优化与辅助价值函数学习解耦。

| PPG 的特征           | 是否具备 | 说明              |
|-------------------|------|-----------------|
| 同策略（On-policy）    | ✅    | 评估策略与目标策略相同。    |
| 异策略（Off-policy）   | ❌    | 评估策略与目标策略不同。    |
| 无模型（Model-free）   | ✅    | 无须预先构建环境动力学模型。  |
| 基于模型（Model-based） | ❌    | 需要使用环境模型训练策略。   |
| 离散动作              | ✅    | 可处理离散动作空间。      |
| 连续动作              | ✅    | 可处理连续动作空间。      |

## 方法

在传统 PPO 中，价值函数被用作策略梯度的基线，并与策略共同训练。
然而，这种耦合可能会限制学习过程的有效性，
因为策略优化可能干扰价值函数的学习，反之亦然。

PPG 通过引入分阶段学习机制解决这一问题，
将策略优化和价值函数学习划分为两个相互独立的阶段。

PPG 包含以下两个阶段：

**策略阶段：**

- 使用 PPO 优化策略，重点是最大化回报。
- 价值函数在策略优化过程中充当基线，但在该阶段不进行训练。

在策略阶段，使用标准 PPO 目标函数优化策略 $\pi_{\theta}$：

$$
L_{PPO}(\theta) = \mathbb{E}[\min{r_t(\theta)A_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t}],
$$

其中：

- $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 为新旧策略的概率比；
- $A_t$ 为优势估计；
- $\epsilon$ 为裁剪参数。

**辅助阶段：**

- 使用辅助损失项训练价值函数。
- 该阶段使价值函数能够更加准确地预测回报，从而提高其作为基线的有效性。

在辅助阶段，通过最小化辅助损失更新价值函数 $V_{\phi}$：

$$
L_{aux}(\phi) = \mathbb{E}[(V_{\phi}(s_t) - R_t)^2 + \beta \cdot L_{consistency}],
$$

其中：

- $R_t$ 为目标回报；
- $L_{consistency}$ 用于约束策略动作与价值预测之间的一致性；
- $\beta$ 为一致性损失项的权重。

上述两个阶段交替进行，
使价值函数和策略能够在不直接相互干扰的情况下更加有效地学习。

## 在 XuanCe 中运行 PPG

在 XuanCe 中运行 PPG 之前，需要先准备一个 conda 环境，并按照
[**安装步骤**](./../../usage/installation.rst)安装 ``xuance``。

### 运行内置示例

完成安装后，可以打开 Python 控制台，并使用以下命令直接运行 PPG：

```python3
import xuance
runner = xuance.get_runner(method='ppg',
                           env='classic_control',  # 可选项：classic_control、box2d、atari。
                           env_id='CartPole-v1',  # 可选项：CartPole-v1、LunarLander-v2、ALE/Breakout-v5 等。
                           is_test=False)
runner.run()  # 也可以使用 runner.benchmark()
```

### 使用自定义配置运行

如需使用不同配置运行 PPG，可以新建一个 ``.yaml`` 文件，例如 ``my_config.yaml``。
然后使用以下代码运行 PPG：

```python3
import xuance as xp
runner = xp.get_runner(method='ppg',
                       env='classic_control',  # 可选项：classic_control、box2d、atari。
                       env_id='CartPole-v1',  # 可选项：CartPole-v1、LunarLander-v2、ALE/Breakout-v5 等。
                       config_path="my_config.yaml",  # 请确保 my_config.yaml 文件的路径正确。
                       is_test=False)
runner.run()  # 也可以使用 runner.benchmark()
```

如需进一步了解配置方法，请参阅
[**配置教程**](./../../api/configs/configuration_examples.rst)。

### 在自定义环境中运行

如需在 XuanCe 尚未包含的自定义环境中运行 PPG，
需要按照[**新环境教程**](./../../usage/custom_env/custom_drl_env.rst)
中的步骤定义新环境。
然后，[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations)
``ppg_myenv.yaml``。

完成上述操作后，可以使用以下代码在自定义环境中运行 PPG：

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import PPG_Agent

configs_dict = get_configs(file_dir="ppg_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # 创建并行环境。
Agent = PPG_Agent(config=configs, envs=envs)  # 创建一个来自 XuanCe 的 PPG 智能体。
Agent.train(configs.running_steps // configs.parallels)  # 对模型进行多个步骤的训练。
Agent.save_model("final_train_model.pth")  # 将模型保存到 model_dir。
Agent.finish()  # 结束训练。
```

## 参考文献

```{code-block} bash
@inproceedings{cobbe2021phasic,
  title={Phasic policy gradient},
  author={Cobbe, Karl W and Hilton, Jacob and Klimov, Oleg and Schulman, John},
  booktitle={International Conference on Machine Learning},
  pages={2020--2027},
  year={2021},
  organization={PMLR}
}
```
