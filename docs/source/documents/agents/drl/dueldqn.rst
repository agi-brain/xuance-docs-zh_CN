Duel DQN
======================

算法描述
----------------------

Dueling DQN算法是对原始DQN算法的另一种改进，
它通过引入双分支结构来分别估计状态价值函数（Value Function）和优势函数（Advantage Function），
从而更有效地学习每个状态下动作的相对优势。
Dueling DQN的主要思想是将原始DQN中的神经网络分为两个分支：
一个分支用于估计状态价值函数V(s)，表示在状态s下不考虑具体动作的期望回报；
另一个分支用于估计优势函数A(s, a)，表示在状态s下采取动作a相对于其他动作的优势。
这两个分支最后会被合并到一个输出层，用于计算状态-动作值函数Q(s, a)。

Dueling DQN的Q值计算方式如下：

.. math:: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a'))).

其中，mean(A(s, a'))表示在状态s下所有动作a'的优势函数均值。
这样的设计可以使得网络更关注动作之间的相对优势，有助于提高学习效率和策略质量。

Dueling network architectures for deep reinforcement learning
论文链接: `Dueling network architectures for deep reinforcement learning 
<http://proceedings.mlr.press/v48/wangf16.pdf>`_

::

    @inproceedings{wang2016dueling,
        title={Dueling network architectures for deep reinforcement learning},
        author={Wang, Ziyu and Schaul, Tom and Hessel, Matteo and Hasselt, Hado and Lanctot, Marc and Freitas, Nando},
        booktitle={International conference on machine learning},
        pages={1995--2003},
        year={2016},
        organization={PMLR}
    }
