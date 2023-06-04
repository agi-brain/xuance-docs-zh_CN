QRDQN
======================

算法描述
----------------------

Quantile DQN，全称Quantile Regression DQN，简称QR-DQN。
是对原始DQN算法的一种改进，旨在更准确地估计状态-动作值函数。

这种方法的主要思想是用一组分位数（quantiles）来表示状态-动作值函数，而不是使用单一的实数值。
这样可以更好地捕捉状态-动作值函数的不确定性，提高强化学习性能。
在Quantile DQN中，状态-动作值函数 :math:`Q(s, a)` 由一组离散的分位数组成。
每个分位数代表一个可能的回报值，具有相应的概率。
为了实现这一目标，Quantile DQN引入了一个神经网络，输出对应每个动作的分位数。
在训练过程中，Quantile DQN使用分位数回归损失（Quantile Regression Loss）来衡量预测分位数与目标分位数之间的差异。

算法出处
----------------------

论文链接: `Dueling network architectures for deep reinforcement learning 
<https://ojs.aaai.org/index.php/AAAI/article/view/11791>`_

::

    @inproceedings{dabney2018distributional,
        title={Distributional reinforcement learning with quantile regression},
        author={Dabney, Will and Rowland, Mark and Bellemare, Marc and Munos, R{\'e}mi},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        volume={32},
        number={1},
        year={2018}
    }
