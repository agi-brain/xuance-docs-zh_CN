TD3
======================

算法描述
----------------------

TD3（Twin Delayed Deep Deterministic Policy Gradient，双延迟深度确定性策略梯度）
算法是一种基于DDPG算法的改进方法，适用于连续动作空间的问题。
TD3在DDPG的基础上引入了三个关键改进：

- 双评论家网络（Twin Critic）
- 延迟策略更新（Delayed Policy Update）
- 目标策略平滑化（Target Policy Smoothing）。

这些改进有助于降低估计偏差，提高学习稳定性和效率。

算法出处
----------------------

**论文链接**:

`Addressing Function Approximation Error in Actor-Critic Methods." In International Conference on Machine Learning  
<http://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf>`_

**论文引用信息**:

::

    @inproceedings{fujimoto2018addressing,
        title={Addressing function approximation error in actor-critic methods},
        author={Fujimoto, Scott and Hoof, Herke and Meger, David},
        booktitle={International conference on machine learning},
        pages={1587--1596},
        year={2018},
        organization={PMLR}
    }
