PDQN
======================

算法描述
----------------------

P-DQN全称为Parameterized Deep Q-Network，是一种基于策略（policy-based）的深度强化学习算法。
P-DQN使用神经网络作为函数逼近器来优化行动的价值，它可用于学习混合动作空间下的最优行动策略。
P-DQN将DDPG算法和DQN相结合，通过DDPG算法确定各个离散动作对应的最优连续参数，
再将这些最优连续参数和状态观测一起输入Q网络，智能体通过比较各个离散动作的价值大小，
选择价值最大的动作作为最优的离散动作。
该离散动作与其对应的最优连续参数共同组成最优参数向量作为智能体的决策信息。

算法出处
----------------------

**论文链接**：
`Parametrized deep q-networks learning: Reinforcement learning with discrete-continuous hybrid action space 
<https://arxiv.org/pdf/1810.06394.pdf>`_

**论文引用信息**：

::

    @article{xiong2018parametrized,
        title={Parametrized deep q-networks learning: Reinforcement learning with discrete-continuous hybrid action space},
        author={Xiong, Jiechao and Wang, Qing and Yang, Zhuoran and Sun, Peng and Han, Lei and Zheng, Yang and Fu, Haobo and Zhang, Tong and Liu, Ji and Liu, Han},
        journal={arXiv preprint arXiv:1810.06394},
        year={2018}
    }
