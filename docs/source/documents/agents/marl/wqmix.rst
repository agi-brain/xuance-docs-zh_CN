WQMIX
======================

算法描述
----------------------

WQMIX（Weighted Q-mixing networks）是在QMIX算法基础上发展而来。
由于QMIX算法的单调性约束，使得它无法解决一些非单调协作问题，因此适用范围受到极大的限制。
官方的WQMXI算法提供了两种实现方式，分别为CWQMIX（Centrally-Weighted Q-mixing networks）和OWQMIX（Optimistically-Weighted Q-mixing Networks）。

CWQMIX算法通过中心加权的方法得到加权QMIX算子，该算法在理论上能够保证将QMIX输出的满足“单调性”约束的函数，
经过权重函数对每个联合动作的加权，映射到非单调值函数，并得到对应的最优策略，从而避免QMIX算法得到的策略陷入局部最优。
OWQMIX（Optimistically-Weighted Q-mixing Networks）采用“乐观”加权的方式，相比CWQMIX在计算上得到了简化。

算法出处
----------------------

**论文链接**:
`Weighted qmix: Expanding monotonic value function factorisation for deep multi-agent reinforcement learning 
<https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf>`_

**引用信息**:

::

    @article{rashid2020weighted,
        title={Weighted qmix: Expanding monotonic value function factorisation for deep multi-agent reinforcement learning},
        author={Rashid, Tabish and Farquhar, Gregory and Peng, Bei and Whiteson, Shimon},
        journal={Advances in neural information processing systems},
        volume={33},
        pages={10199--10210},
        year={2020}
    }
