# DrQ: Data-Regularized Q-Learning

**论文链接:** [**ArXiv**](https://arxiv.org/abs/2004.13649)

Data-Regularized Q-learning（简称 **DrQ**）是一种基于数据增强（data augmentation）的**离策略深度强化学习算法**，用于提升在基于像素的控制任务中的样本效率与泛化能力。  
它通过在训练过程中对输入观测应用多种随机变换，来解决传统 Q-learning 在高维视觉输入下容易出现的**过拟合与泛化不良**问题。

下表总结了 DrQ 算法的主要特征：

| 特征类别 | 值 | 描述 |
|-----------|----|------|
| On-policy | ❌ | 评估策略与目标策略相同 |
| Off-policy | ✅ | 评估策略与目标策略不同 |
| Model-free | ✅ | 无需环境动力学模型 |
| Model-based | ❌ | 需构建环境模型以训练策略 |
| Discrete Action | ✅ | 处理离散动作空间 |
| Continuous Action | ❌ | 不适用于连续动作空间 |

---

## 算法描述（Algorithm Description）

DrQ 针对高维视觉输入下 Q-learning 的困难问题，提出了利用**数据增强**的解决思路。  
核心思想是在训练阶段对输入观测施加多种随机变换，以正则化学习过程并提升泛化能力。  
这种方法能够有效降低模型对特定视觉模式的过拟合，使学习到的策略对输入变化更加稳健。

其主要洞见在于：通过数据增强，Q 网络被迫在同一观测的不同增强版本上输出一致的 Q 值，从而学习到**与任务相关的语义信息**，而非表面的像素细节。

---

## 网络结构（Network Architecture）

DrQ 智能体使用卷积神经网络（CNN）作为 Q 网络的骨干结构。  
该结构通常包含多个卷积层与全连接层。虽然具体架构会随环境与任务而异，但核心目标一致——  
**通过数据增强作为正则项，从像素输入中学习 Q 值函数。**

---

## 实现细节（Implementation Details）

### 数据增强（Data Augmentation）

DrQ 在训练过程中对输入观测进行多种随机数据增强。常见的增强方式包括：
- 随机裁剪（Random cropping）  
- 颜色扰动（Color jittering）  
- 随机灰度化（Random grayscale conversion）  
- 水平翻转（Horizontal flipping）  

这些增强样本在训练时动态生成，并用于更新 Q 网络。

---

### 训练流程（Training Process）

DrQ 的训练步骤如下：
1. 从环境中收集经验数据；
2. 对观测施加多种随机增强；
3. 使用增强后的观测更新 Q 网络；
4. 定期同步目标网络。

Q 网络使用带有增强观测的标准时间差（TD）误差损失进行训练：

$$
\mathcal{L}_{\text{Q}} = \mathbb{E}[(Q(s,a) - (r + \gamma \max_{a'} Q'(s',a')))^2]
$$

其中，$Q'$ 为目标网络，$\gamma$ 为折扣因子，$r$ 为即时奖励。

为了增强鲁棒性，DrQ 对每个观测生成多种增强版本，并将其平均：

$$
\bar{Q}(s,a) = \frac{1}{K} \sum_{k=1}^{K} Q(s_k,a)
$$

其中，$K$ 为增强次数，$s_k$ 为第 $k$ 个增强观测版本。

---

## 关键特性（Key Features）

### 提升泛化能力（Improved Generalization）

通过数据增强，DrQ 显著提高了 Q 网络的泛化能力。  
模型被迫在不同增强版本上输出一致的 Q 值，从而专注于观测的语义内容而非低级视觉特征。

### 样本效率（Sample Efficiency）

DrQ 通过更高效地利用已有样本提升了样本效率。  
由于每个观测可生成多个增强样本，训练数据集在不增加环境交互次数的前提下得到有效扩展。

### 鲁棒性（Robustness）

数据增强使得学习到的策略对输入变化更加稳健，特别适用于视觉输入随光照、角度、噪声等因素变化的实际环境。

---

## 优势（Advantages）

1. **实现简单**：DrQ 结构清晰，可轻松嵌入现有 Q-learning 框架中；  
2. **有效正则化**：数据增强提供了天然的正则项，有助于防止过拟合；  
3. **性能显著提升**：在基于像素的控制任务中，DrQ 明显优于传统 Q-learning 方法。

---

## 应用场景（Application Scenarios）

DrQ 尤其适用于以下任务类型：
- 基于像素输入的控制任务（如 Atari 游戏环境）；
- 对样本效率要求高的场景；
- 需要对视觉扰动具备鲁棒性的应用。

---

## 算法流程（Algorithm）

DrQ 的完整训练流程如下图所示：

```{eval-rst}
.. image:: ./../../../../_static/figures/pseucodes/pseucode-DrQ.png
    :width: 80%
    :align: center
```

---

## 在 XuanCe 中运行 DrQ（Run DrQ in XuanCe）

在运行 DrQ 之前，需要准备 Conda 环境并按照  
[**安装步骤**](./../../../usage/installation.rst#install-xuance) 安装 ``xuance``。

### 运行内置示例（Run Built-in Demos）

安装完成后，可在 Python 控制台中直接运行：

```python3
import xuance
runner = xuance.get_runner(method='drq',
                           env='atari',
                           env_id='ALE/Breakout-v5',
                           is_test=False)
runner.run()  # 或 runner.benchmark()
```

---

### 使用自定义配置文件（Run With Self-defined Configs）

若希望使用自定义配置运行 DrQ，可新建 ``.yaml`` 文件（例如 ``my_config.yaml``），  
然后执行以下代码：

```python3
import xuance as xp
runner = xp.get_runner(method='drq',
                       env='atari',
                       env_id='ALE/Breakout-v5',
                       config_path="my_config.yaml",
                       is_test=False)
runner.run()  # 或 runner.benchmark()
```

更多关于配置文件的内容可参见  
[**配置教程**](./../../configs/configuration_examples.rst)。

---

## 参考文献（Citations）

```bash
@article{yarats2021image,
  title={Image augmentation is all you need: Regularizing deep reinforcement learning from pixels},
  author={Yarats, Denis and Kostrikov, Ilya and Fergus, Rob},
  journal={arXiv preprint arXiv:2004.13649},
  year={2021}
}
```

---

## API 接口（APIs）

### PyTorch

```{eval-rst}
.. automodule:: xuance.torch.agents.contrastive_unsupervised_rl.drq_agent
    :members:
    :undoc-members:
    :show-inheritance:
```

### TensorFlow2

```{eval-rst}
.. automodule:: xuance.tensorflow.agents.contrastive_unsupervised_rl.drq_agent
    :members:
    :undoc-members:
    :show-inheritance:
```

### MindSpore

```{eval-rst}
.. automodule:: xuance.mindspore.agents.contrastive_unsupervised_rl.drq_agent
    :members:
    :undoc-members:
    :show-inheritance:
```
