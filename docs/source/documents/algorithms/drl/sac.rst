SAC
======================

算法描述
----------------------

SAC算法全称为Soft actor-critic based on maximum entropy，
是一种基于执行器-评价器（actor-critic based）的深度强化学习算法。
SAC分别使用不同的神经网络来拟合智能体的策略和值函数，其中拟合策略的神经网络输出的是一个高斯分布，
主要用于解决连续动作空间下的控制问题。SAC算法与基础的Actor-critic算法不同之处在于，
它在最大化累计奖励的同时考虑最大化策略的熵。
具体的，SAC算法在策略的目标函数上增加策略的期望熵，增加这一目标的优点在于能够鼓励智能体探索和可以找到多个接近最优的路径。

在本算法库中，SAC的前向传播分为representation，eval_policy, eval_Q三个部分。
representation由包含单隐层的MLP构成，输入为智能体观测的状态信息，输出为256维的隐状态信息。
eval_a 模块的输入为representation的输出，再次经过单隐层的MLP网络输出动作的高斯分布，智能体通过在高斯分布中采样选择最后施加于环境中的动作。
Eval_Q模块的输入为representation的输出，再次经过单隐层的MLP网络输出价值函数。
此外，SAC还包含target_a和target_Q作为目标动作网络和目标Q网络，目标网络与在线网络结构系统相同，参数和在线网络保持周期性一致。

算法出处
----------------------

**论文链接**:
`Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor 
<http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf>`_

**论文引用信息**:

::

    @inproceedings{haarnoja2018soft,
        title={Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor},
        author={Haarnoja, Tuomas and Zhou, Aurick and Abbeel, Pieter and Levine, Sergey},
        booktitle={International conference on machine learning},
        pages={1861--1870},
        year={2018},
        organization={PMLR}
    }
