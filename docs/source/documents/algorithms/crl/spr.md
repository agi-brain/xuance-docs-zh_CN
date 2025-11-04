# SPR: Self-Predictive Representations for Reinforcement Learning

**论文链接:** [**ArXiv**](https://arxiv.org/abs/2007.05929)

Self-Predictive Representations（简称 **SPR**）是一种**离策略深度强化学习（off-policy deep reinforcement learning）算法**，  
通过在潜空间（latent space）中预测未来状态来学习表示。  
SPR 结合了对比学习方法（如 CURL）的优点与预测模型的思想，使其能够更高效地从高维视觉输入中学习。  
它解决了传统强化学习在像素级控制任务中样本效率低下的问题，通过学习能够捕捉时间依赖关系的丰富状态表示，提高了训练效率。

---

| SPR 算法特征 | 值 | 描述 |
|---------------|----|------|
| On-policy | ❌ | 评估策略与目标策略相同 |
| Off-policy | ✅ | 评估策略与目标策略不同 |
| Model-free | ✅ | 无需环境动态模型 |
| Model-based | ❌ | 需要环境模型来训练策略 |
| Discrete Action | ✅ | 处理离散动作空间 |
| Continuous Action | ❌ | 不支持连续动作空间 |

---

## 算法描述（Algorithm Description）

SPR 针对强化学习从像素学习时的样本效率问题，通过学习**时间预测表示（temporally predictive representations）**来改进。  
其主要思想是训练卷积编码器（CNN encoder），使其能够预测自身潜在表示的未来若干步。  
这种方法促使编码器学习到能够反映环境动态的核心特征，同时对任务无关的细节保持不敏感。

核心洞见在于：通过在潜空间中预测未来状态，SPR 能够学习到既**具有预测性又稳定**的表示。  
这使得智能体能够更高效地利用过往经验，从而提升策略学习的样本效率。

---

## 网络结构（Network Architecture）

SPR 智能体采用卷积神经网络（CNN）作为表示网络的骨干结构。  
该结构通常包含多个卷积层与全连接层，编码器将观测值映射到潜在表示空间（latent representations），  
随后这些表示被 Q 网络用于估计 Q 值。

SPR 在编码器的基础上引入了**状态转移模型（transition model）**，  
该模型输入当前潜在状态与动作序列，预测未来潜在状态序列，从而在潜空间中建模环境动态。

---

## 实现细节（Implementation Details）

### 智能体实现（Agent Implementation）

```python
class SPR_Agent(OffPolicyAgent):
    def __init__(self, config: Namespace, envs: Union[DummyVecEnv, SubprocVecEnv]):
        super(SPR_Agent, self).__init__(config, envs)
        self._init_exploration_params(config)
        
        self.policy = self._build_policy()  # 构建策略
        self.memory = self._build_memory()  # 构建经验池
        self.learner = self._build_learner(config, self.policy)  # 构建学习器
        self.transform = SPR_Augmentations.get_transform(self.observation_space.shape[-1])

    def _init_exploration_params(self, config: Namespace):
        self.e_greedy = config.start_greedy
        self.e_greedy_decay = (config.start_greedy - config.end_greedy) / (config.decay_step_greedy / self.n_envs)
```

---

### 编码器与状态转移模型（Encoder and Transition Model）

SPR 的编码器将观测映射到潜在表示空间，  
而转移模型（transition model）则根据当前表示与动作预测未来潜在表示。

```python
class SPR_Encoder(nn.Module):
    # SPR 编码器（基于 CNN 的结构）
    def __init__(self, observation_space: Space, config: Namespace, device: str):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.LayerNorm(512)
        )

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return self.net(x)
```

---

### 训练流程（Training Process）

SPR 的训练步骤如下：

1. 从环境中采集交互经验；  
2. 对观测数据进行数据增强（data augmentations）；  
3. 使用**对比损失（contrastive loss）**与**预测损失（predictive loss）**共同更新表示网络；  
4. 使用学得的表示更新 Q 网络；  
5. 定期更新目标网络参数。

SPR 的总损失函数由对比项与预测项组成：

$$
\mathcal{L}_{\text{SPR}} = \mathcal{L}_{\text{contrastive}} + \lambda \mathcal{L}_{\text{predictive}}
$$

其中，$\lambda$ 为平衡两种损失的权重因子。

预测损失（predictive loss）用于约束转移模型准确预测未来潜在状态：

$$
\mathcal{L}_{\text{predictive}} = \sum_{k=1}^{K} \| \hat{z}_{t+k} - z_{t+k} \|_2^2
$$

其中 $\hat{z}_{t+k}$ 为时间步 $t+k$ 的预测潜在表示，$z_{t+k}$ 为真实潜在表示，$K$ 为预测步数。

---

## 关键特性（Key Features）

### 时间预测学习（Temporal Predictive Learning）

SPR 通过预测未来潜在状态进行学习，  
促使编码器捕捉环境的时间动态特征，从而获得更具表达力的表示。

### 样本效率（Data Efficiency）

通过学习捕捉时间依赖的丰富表示，SPR 在样本利用率上显著优于传统 Q-learning。  
智能体能够更好地理解环境随时间的变化，从而高效使用已有经验。

### 鲁棒性（Robustness）

对比学习与预测学习的结合使得 SPR 学到的表示在视觉扰动下更稳定。  
这对于实际应用中存在光照、角度、噪声变化的视觉任务尤为重要。

---

## 优势（Advantages）

1. **高样本效率**：SPR 通过预测性表示显著提升了样本利用效率；  
2. **时间理解能力**：预测组件使智能体具备对时间动态的理解；  
3. **鲁棒表示**：对比与预测结合使得表示更加稳健。

---

## 应用场景（Application Scenarios）

SPR 特别适用于以下场景：
- 基于像素输入的控制任务；  
- 对样本效率要求高的环境；  
- 需要理解时间动态的强化学习应用。

---

## 算法（Algorithm）

SPR 的完整训练流程如下所示：

```{eval-rst}
.. image:: ./../../../../_static/figures/pseucodes/pseucode-SPR.png
    :width: 80%
    :align: center
```

---

## 框架（Framework）

SPR 在 XuanCe 框架中的智能体–环境交互如下图所示：

```{eval-rst}
.. image:: ./../../../../_static/figures/algo_framework/spr_framework.png
    :width: 100%
    :align: center
```

---

## 在 XuanCe 中运行 SPR（Run SPR in XuanCe）

在运行 SPR 之前，需要准备 Conda 环境并按照  
[**安装步骤**](./../../../usage/installation.rst#install-xuance) 安装 ``xuance``。

### 运行内置示例（Run Built-in Demos）

安装完成后，可直接运行以下命令：

```python3
import xuance
runner = xuance.get_runner(method='spr',
                           env='atari',
                           env_id='ALE/Breakout-v5',
                           is_test=False)
runner.run()  # 或 runner.benchmark()
```

---

### 使用自定义配置文件（Run With Self-defined Configs）

若需自定义配置，可新建 ``my_config.yaml`` 文件，然后执行：

```python3
import xuance as xp
runner = xp.get_runner(method='spr',
                       env='atari',
                       env_id='ALE/Breakout-v5',
                       config_path="my_config.yaml",
                       is_test=False)
runner.run()  # 或 runner.benchmark()
```

更多内容请参考  
[**配置教程**](./../../configs/configuration_examples.rst)。

---

## 参考文献（Citations）

```bash
@inproceedings{raileanu2021spr,
  title={Self-predictive representation learning},
  author={Raileanu, Robert and Fergus, Rob},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

---

## API 接口（APIs）

### PyTorch

```{eval-rst}
.. automodule:: xuance.torch.agents.contrastive_unsupervised_rl.spr_agent
    :members:
    :undoc-members:
    :show-inheritance:
```

### TensorFlow2

```{eval-rst}
.. automodule:: xuance.tensorflow.agents.contrastive_unsupervised_rl.spr_agent
    :members:
    :undoc-members:
    :show-inheritance:
```

### MindSpore

```{eval-rst}
.. automodule:: xuance.mindspore.agents.contrastive_unsupervised_rl.spr_agent
    :members:
    :undoc-members:
    :show-inheritance:
```
