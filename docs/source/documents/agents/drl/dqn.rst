DQN
===============================

算法描述
-------------------------------

DQN算法全称为Deep Q-Network，是一种基于值函数（value-based）的深度强化学习算法。
DQN使用神经网络作为函数逼近器来优化行动的价值，它可用于学习离散动作空间下的最优行动策略，
近年来在强化学习领域取得了很多令人瞩目的成果。DQN在Q-learning算法的基础上，
通过使用深度神经网络来替代Q表进行状态和行为的函数逼近，有效提升了深度强化学习中学习到的动作决策的效果，
可用于学习在复杂场景下的最优决策策略。

在玄策中，DQN的前向传播分为representation和eval_Q两个部分。
representation由包含单隐层的多层感知器（Multi-Layer Perception, MLP）构成，
输入为智能体观测的状态信息，输出为256维的隐状态信息。Eval_Q模块的输入为representation的输出，
再次经过单隐层的MLP网络输出两个动作的价值。智能体通过比较两个动作的价值大小，
选择价值最大对应的动作作为决策信息。此外，DQN还包含target_Q作为目标Q网络，其结构和eval_Q相同，
参数和eval_Q保持周期性一致。

算法出处
-------------------------------

**论文链接**: `Human-level control through deep reinforcement learning 
<https://www.nature.com/articles/nature14236/>`_

**论文引用信息**:

::

    @article{mnih2015human,
      title={Human-level control through deep reinforcement learning},
      author={Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A and Veness, Joel and Bellemare, Marc G and Graves, Alex and Riedmiller, Martin and Fidjeland, Andreas K and Ostrovski, Georg and others},
      journal={nature},
      volume={518},
      number={7540},
      pages={529--533},
      year={2015},
      publisher={Nature Publishing Group}
    }

