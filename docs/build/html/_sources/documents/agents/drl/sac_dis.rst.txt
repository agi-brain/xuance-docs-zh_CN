SAC
======================

算法描述
----------------------

SAC-Dis算法全称为soft actor-critic based on maximum entropy with discrete action，
是一种基于执行器-评价器（actor-critic based）的深度强化学习算法。
SACDIS分别使用不同的神经网络来拟合智能体的策略和值函数，其中拟合策略的神经网络输出的是一个分类分布，
主要用于解决离散动作空间下的控制问题。
SACDIS 算法与SAC算法不同之处在于其策略网络是一个分类分布，用来解决离散动作空间控制问题。
SACDIS算法仍然要在最大化累计奖励的同时对离散策略的熵进行最大化。

在本算法库中，SACDIS的前向传播分为representation，eval_policy, eval_Q三个部分。
representation由包含单隐层的多层感知器MLP构成，输入为智能体观测的状态信息，输出为256维的隐状态信息。
eval_a 模块的输入为representation的输出，再次经过单隐层的MLP网络输出动作的高斯分布，
智能体通过在分类分布中选择最后施加于环境中的动作。
Eval_Q模块的输入为representation的输出，再次经过单隐层的MLP网络输出动作的价值。
此外，SACDIS还包含target_Q作为目标Q网络，目标网络与在线网络结构系统相同，参数和在线网络保持周期性一致。

算法出处
----------------------

论文链接：

- 
`Soft actor-critic for discrete action settings 
<https://arxiv.org/pdf/1910.07207.pdf>`_

论文引用信息：

::

    @article{christodoulou2019soft,
        title={Soft actor-critic for discrete action settings},
        author={Christodoulou, Petros},
        journal={arXiv preprint arXiv:1910.07207},
        year={2019}
    }
