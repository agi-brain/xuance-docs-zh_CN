PPO
======================

算法描述
----------------------

PPO-CLIP（Proximal Policy Optimization with Clipped Objective，带截断目标的近端策略优化）
是一种基于策略梯度的深度强化学习算法，用于解决策略优化问题。
PPO-CLIP的主要优点是稳定性和样本效率，它通过限制策略更新幅度来避免过大的策略改进，从而确保在训练过程中策略的稳定性。
PPO-CLIP算法的核心思想是在策略更新时限制策略概率比率的变化范围。

PPO-KL（Proximal Policy Optimization with Kullback-Leibler Divergence Constraint，带Kullback-Leibler散度约束的近端策略优化）
是一种基于策略梯度的深度强化学习算法，用于解决策略优化问题。
PPO-KL通过限制策略更新的幅度来确保训练过程中策略的稳定性，从而提高学习性能。
PPO-KL算法的核心思想是在策略更新时限制新策略与旧策略之间的Kullback-Leibler (KL)散度。
为了实现这一目标，PPO-KL引入了一个带KL散度约束的目标函数。

算法出处
----------------------

论文链接：

- `Proximal policy optimization algorithms 
<https://arxiv.org/pdf/1707.06347.pdf>`_

论文引用信息：

::

    @article{schulman2017proximal,
        title={Proximal policy optimization algorithms},
        author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
        journal={arXiv preprint arXiv:1707.06347},
        year={2017}
    }
