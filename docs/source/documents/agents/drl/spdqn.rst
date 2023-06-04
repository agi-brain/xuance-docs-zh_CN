SPDQN
======================

算法描述
----------------------

SP-DQN：SP-DQN全称为Separate-Parameterized Deep Q-Network, 是一种基于策略(policy-based)的深度强化学习算法。
MP-DQN使用神经网络作为函数逼近器来优化行动的价值，它可用于学习混合动作空间下的最优行动策略。
MP-DQN将DDPG算法和DQN相结合，通过DDPG算法确定各个离散动作对应的最优连续参数, 
考虑到全通道输入所有离散动作对应的最优连续参数进Q网络影响了不同离散动作对应Q值的梯度，
在P-DQN算法的基础上通过将各个离散动作对应的最优连续参数分离并单独输入Q网络的方式计算不同离散动作对应的Q值，
智能体通过比较各个离散动作的价值大小，选择价值最大的动作作为最优的离散动作。
该离散动作与其对应的最优连续参数共同组成最优参数向量作为智能体的决策信息。

算法出处
----------------------

论文链接：

- `Multi-pass q-networks for deep reinforcement learning with parameterised action spaces 
<https://arxiv.org/pdf/1905.04388.pdf>`_

论文引用信息：

::

    @article{bester2019multi,
        title={Multi-pass q-networks for deep reinforcement learning with parameterised action spaces},
        author={Bester, Craig J and James, Steven D and Konidaris, George D},
        journal={arXiv preprint arXiv:1905.04388},
        year={2019}
    }
