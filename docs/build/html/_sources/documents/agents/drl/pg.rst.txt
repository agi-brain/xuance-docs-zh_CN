Policy Gradient
======================

算法描述
----------------------
PG（Policy Gradient）算法是一种基于策略梯度的深度强化学习算法，用于解决策略优化问题。
PG算法的主要思想是直接对策略进行优化，即通过最大化累积奖励来寻找最优策略。
PG算法不需要对状态-动作值函数（Q函数）进行估计，而是直接优化策略参数，从而避免了Q函数估计误差的传递和累积。
PG算法的主要更新过程包括两个步骤：采样和梯度上升。在采样过程中，PG算法使用当前策略与环境交互，得到一系列轨迹数据。
在梯度上升过程中，PG算法使用策略梯度定理，通过对每条轨迹上的动作概率进行梯度上升，来更新策略参数。
PG算法还可以使用基线函数来减少方差，从而提高更新效率。


算法出处
----------------------

论文链接：

- `A natural policy gradient 
<https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf>`_

论文引用信息：

::

    @article{kakade2001natural,
        title={A natural policy gradient},
        author={Kakade, Sham M},
        journal={Advances in neural information processing systems},
        volume={14},
        year={2001}
    }
