# 独立执行器-评判器 (IAC)

**论文链接:** [**https://ojs.aaai.org/index.php/AAAI/article/view/11794**](https://ojs.aaai.org/index.php/AAAI/article/view/11794).

在多智能体强化学习（MARL）领域，独立执行器-评判器（IAC）是一种基础且直观的分布式学习范式。它旨在解决由多个智能体组成的协作系统中的决策问题。IAC 算法的核心思想是将单智能体环境中的执行器-评判器（AC）框架直接推广到多智能体场景。每个智能体独立运行完整的 AC 架构，并通过局部经验进行学习。

下表列出了 IAC 算法的一些一般特性：

| IAC 的特性                                       | 值   | 描述                                                                                                 |
|--------------------------------------------------|------|-----------------------------------------------------------------------------------------------------|
| 完全去中心化                                        | ✅    | 智能体之间没有通信。                                                                                       |
| 完全中心化                                          | ❌    | 智能体将所有信息发送给中央控制器，控制器将为所有智能体做出决策。                                                        |
| 中心化训练与去中心化执行（CTDE）                            | ❌    | 在训练中使用中央控制器，在执行中放弃。                                                                          |
| 同策略                                               | ✅    | 评估策略与目标策略相同。                                                                                     |
| 异策略                                               | ❌    | 评估策略与目标策略不同。                                                                                     |
| 无模型                                               | ✅    | 不需要准备环境动力学模型。                                                                                   |
| 基于模型                                             | ❌    | 需要环境模型来训练策略。                                                                                     |
| 离散动作                                             | ✅    | 处理离散动作空间。                                                                                         |
| 连续动作                                             | ❌    | 处理连续动作空间。                                                                                         |

## 背景

IAC 算法的核心思想源于单智能体执行器-评判器框架的直接推广，其设计意图是为多智能体系统提供一种轻量级的学习解决方案，无需跨智能体协调。在完全协作的多智能体任务中，IAC 算法将每个智能体视为独立的学习个体。通过为每个智能体配置专用的执行器（Executor）和评判器（Evaluator）组件，它实现了分布式决策和价值评估过程。这种设计严格遵循"完全去中心化"范式，在训练阶段不需要全局信息共享，在执行阶段也不需要跨智能体通信，大大降低了算法的实现复杂度。

## 核心组件与运行机制

### 独立执行器网络

每个智能体的执行器网络仅基于其自身的动作-观测历史 $\tau^a \in T \equiv (Z \times U)^*$ 输出动作分布，即策略 $\pi^a(u^a|\tau^a) : T \times U \to [0, 1]$。在执行阶段，每个执行器完全独立运行，仅依赖局部可观测信息做出决策，这符合实际场景中智能体的感知约束。

### 专用评判器网络

每个智能体的评判器网络负责评估其自身策略的价值。它通常以局部观测 $z \in Z$ 和执行的动作 $u^a \in U$ 作为输入，输出状态价值 $V(s)$ 或优势函数 $A(s, a)$，为相应的执行器网络提供更新依据。

### 参数更新机制

IAC 采用经典的执行器-评判器更新规则。评判器网络通过时序差分误差（TD-error）优化价值估计的准确性，执行器网络基于评判器提供的价值信号执行策略梯度上升，以最大化期望累积奖励。在参数共享场景中，多个智能体可以重用同一组网络参数，同时保持其独立的学习过程。

### 形式化表达

结合 COMA 论文中定义的随机博弈 $G = \langle S, U, P, r, Z, O, n, \gamma \rangle$，IAC 算法的运行场景可以形式化描述如下：$n$ 个智能体 $a \in A \equiv \{1, \dots, n\}$ 处于环境状态 $s \in S$ 中，根据观测函数 $O(s, a) : S \times A \to Z$ 获取局部观测，通过独立策略选择动作形成联合动作 $\mathbf{u} \in \mathbf{U} \equiv U^n$，并根据共享奖励函数 $r(s, \mathbf{u}) : S \times \mathbf{U} \to \mathbb{R}$ 获得全局奖励。由于缺乏全局信息融合机制，每个智能体的评判器网络只能基于局部信号近似评估全局奖励与其自身动作之间的关联性。

## 局限性

### 信用分配问题导致的梯度混淆

在完全协作的多智能体任务中，奖励信号通常是全局共享的，单个智能体的动作价值无法通过全局奖励直接区分。IAC 算法采用"各自为政"的价值评估模式，所有智能体的评判器网络都基于相同的全局奖励进行更新。这导致执行器网络接收到的梯度信号无法准确反映其自身动作对团队性能的实际贡献，造成"平均分配"的训练困境。

COMA 论文进一步指出，当智能体数量增加时，这种梯度混淆问题会显著加剧。由于每个智能体的梯度更新受到所有其他智能体动作的干扰，参数共享的 IAC 模型会产生严重的噪声梯度，使得训练过程难以收敛到最优联合策略。

### 缺乏交互建模导致的次优解

IAC 算法的核心假设是"智能体行为可以视为环境动力学的一部分"，即每个智能体将其他智能体的动作等同于环境噪声，不需要主动建模交互关系。这一假设在动态和复杂的协作场景中具有明显的缺陷：智能体无法预测队友的动作意图，导致联合动作的协调性显著下降。

例如，在需要分工协作的多智能体任务中，IAC 算法可能导致多个智能体重复执行相同的有效动作或错过关键的协作步骤。COMA 论文通过对比实验证实，缺乏交互建模的这一特性使 IAC 算法在大多数协作任务中只能收敛到次优解，其性能随着智能体数量的增加而急剧下降。

### 观测局限性导致的价值估计偏差

IAC 算法的评判器网络仅依赖智能体的局部观测进行价值评估。在部分可观测场景中，局部观测无法充分反映环境的真实状态 $s \in S$。这种部分可观测性导致评判器网络的价值估计出现系统性偏差，进而误导执行器网络的策略更新方向。

与 COMA 算法采用的中心化评判器（以全局状态或联合动作-观测历史作为输入）相比，IAC 的去中心化评判器缺乏全局信息校正机制，难以准确估计联合动作价值 $Q(s, \mathbf{u})$。这导致优势函数计算中的误差不断累积，最终影响训练稳定性。

## 在 XuanCe 中运行 IAC

在 XuanCe 中运行 IAC 之前，您需要准备 conda 环境并按照 
[**安装步骤**](./../../usage/installation.rst#install-xuance) 安装 ``xuance``。

### 运行内置演示

完成安装后，您可以打开 Python 控制台并使用以下命令直接运行 IAC：

```python3
import xuance
runner = xuance.get_runner(method='iac',
                    env='mpe',  
                    env_id='simple_spread_v3',  
                    is_test=False)
runner.run() 
```
### 使用自定义配置运行

如果您想使用不同的配置运行 IAC，可以创建一个新的 ``.yaml`` 文件，例如 ``my_config.yaml``。
然后，通过以下代码块运行 IAC：

```python3
import xuance
runner = xuance.get_runner(method='iac',
                       env='mpe', 
                       env_id='simple_spread_v3',  
                       config_path="my_config.yaml",  # my_config.yaml 文件的路径应该是正确的。
                       is_test=False)
runner.run()
```
### 使用自定义环境运行

如果您想在 XuanCe 中未包含的自己的环境中运行 XuanCe 的 IAC， 
您需要按照
[**新环境教程**](./../../usage/custom_env/custom_drl_env.rst) 中的步骤定义新环境。
然后，[**准备配置文件**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``iac_myenv.yaml``。

之后，您可以使用以下代码在自己的环境中运行 IAC：

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import IAC_Agents

configs_dict = get_configs(file_dir="iac_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs) 
Agent = IAC_Agents(config=configs, envs=envs) 
Agent.train(configs.running_steps // configs.parallels)  
Agent.save_model("final_train_model.pth") 
Agent.finish()  # 完成训练。
```


## 引用

```{code-block} bash
@inproceedings{foerster2018counterfactual,
  title={Counterfactual multi-agent policy gradients},
  author={Foerster, Jakob and Farquhar, Gregory and Afouras, Triantafyllos and Nardelli, Nantas and Whiteson, Shimon},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={32},
  number={1},
  year={2018}
}
```

