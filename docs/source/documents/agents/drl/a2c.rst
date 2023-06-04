A2C
======================

算法描述
----------------------

A2C（Advantage Actor-Critic，优势执行-评价器）算法是一种基于策略梯度的深度强化学习算法，
它使用了执行器-评价器（Actor-Critic）框架。在这个框架中，执行器（Actor）负责学习一个策略（policy），
即在给定状态下选择动作的概率分布；评价器（Critic）负责学习一个值函数（value function），即估计状态或状态-动作对的价值。
A2C算法结合了策略优化和值函数逼近，旨在平衡探索与利用，提高学习性能。
A2C算法的一大优点是能够在连续或离散的动作空间中进行有效学习。
此外，它可以很自然地与其他强化学习技术（如经验回放、模型预测控制等）结合使用。

算法出处
----------------------

**论文链接**：
`Asynchronous methods for deep reinforcement learning 
<http://proceedings.mlr.press/v48/mniha16.pdf>`_

**论文引用信息**：

::

    @inproceedings{mnih2016asynchronous,
        title={Asynchronous methods for deep reinforcement learning},
        author={Mnih, Volodymyr and Badia, Adria Puigdomenech and Mirza, Mehdi and Graves, Alex and Lillicrap, Timothy and Harley, Tim and Silver, David and Kavukcuoglu, Koray},
        booktitle={International conference on machine learning},
        pages={1928--1937},
        year={2016},
        organization={PMLR}
    }

