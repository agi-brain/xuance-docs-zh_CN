DDPG
======================

算法描述
----------------------

DDPG（Deep Deterministic Policy Gradient，深度确定性策略梯度）
算法是一种基于执行-评价（Actor-Critic）框架的深度强化学习算法，适用于连续动作空间的问题。
DDPG结合了深度学习和确定性策略梯度方法，以实现高维状态空间和连续动作空间下的高效学习。
DDPG算法的主要步骤如下：
#. 初始化演员网络（Actor）和评论家网络（Critic）以及他们的目标网络。
#. 使用演员网络在环境中采集轨迹（trajectory）。
#. 根据采集到的轨迹，使用评论家网络计算状态-动作对的Q值。
#. 使用确定性策略梯度方法更新演员网络的参数。具体来说，通过最大化期望Q值来更新策略。
#. 使用均方误差损失（Mean Squared Error Loss）更新评论家网络的参数，使其能够更准确地估计状态-动作对的价值。
#. 使用软更新策略更新目标网络的参数，确保学习过程的稳定性。软更新通过线性插值的方式平滑地更新目标网络的参数，而非直接复制。

算法出处
----------------------

论文链接：

- 
`Continuous control with deep reinforcement learning 
<https://arxiv.org/pdf/1509.02971.pdf>`_

论文引用信息：

::

    @article{lillicrap2015continuous,
        title={Continuous control with deep reinforcement learning},
        author={Lillicrap, Timothy P and Hunt, Jonathan J and Pritzel, Alexander and Heess, Nicolas and Erez, Tom and Tassa, Yuval and Silver, David and Wierstra, Daan},
        journal={arXiv preprint arXiv:1509.02971},
        year={2015}
    }
