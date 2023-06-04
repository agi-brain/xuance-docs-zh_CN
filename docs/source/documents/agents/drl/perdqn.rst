Per DQN
======================

算法描述
----------------------

Per DQN: Prioritized Replay Buffer DQN（基于优先级的经验回放的DQN）是对原始DQN算法的一种改进，
旨在更高效地从经验回放缓存中采样有价值的经验，从而提高学习效率。
这种方法的主要思想是为经验回放缓冲区中的每个转换分配一个优先级，并根据优先级进行采样，而不是进行均匀随机采样。
在Prioritized Replay Buffer DQN中，每个经验元组
（状态 :math:`s`、动作 :math:`a`、奖励 :math:`r`、下一个状态 :math:`s'`）的优先级计算方法如下：

.. math:: p = |\delta| + \epsilon，

其中，:math:`\delta` 表示该经验元组的TD误差（Temporal Difference Error），
即预测Q值与实际Q值之间的差值；:math:`\epsilon` 是一个很小的正数，用于确保每个经验元组都有一定的被采样概率。

算法出处
---------------------

论文链接: `Prioritized experience replay 
<https://arxiv.org/pdf/1511.05952>`_

论文引用信息:

::

    @article{schaul2015prioritized,
        title={Prioritized experience replay},
        author={Schaul, Tom and Quan, John and Antonoglou, Ioannis and Silver, David},
        journal={arXiv preprint arXiv:1511.05952},
        year={2015}
    }
