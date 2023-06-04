MATD3
======================

算法描述
----------------------

MATD3算法全称为multi-agent delayed deterministic policy gradient，
是一种基于执行器-评价器（actor-critic based）的多智能体深度强化学习算法。
MATD3将单智能体领域的TD3延伸到多智能体领域，其主要结构仍然采取集中训练和独立执行的形式，
即每个智能体的值函数输入空间不仅包括自身的观测和动作，也包括其他所有智能体的观测和动作。
但每个智能体的策略输入空间也只有自身的观测数据。
MATD3是MADDPG的优化版算法, 其使用两套网络(Twin) 表示不同的Q（Q_A和Q_B）值，
通过选取最小的那个作为更新的目标，从而抑制持续地过高估计。

在本算法库中，MATD3应用于三个智能体强化学习任务场景。
每个智能体的前向传播分为representation，eval_policy, eval_Q_A和eval_Q_B四个部分。
representation由包含单隐层的多层感知器MLP构成，输入为智能体观测的状态信息，输出为32维的隐状态信息。
eval_a 模块的输入为representation的输出，再次经过单隐层的MLP网络输出动作的高斯分布，
智能体通过在高斯分布中采样选择最后施加于环境中的动作。Eval_Q_A和Eval_Q_B模块的输入为representation的输出，
再次经过单隐层的MLP网络值函数。此外，MATD3中的每个智能体还包含target_a和target_Q作为目标动作网络和目标Q网络，
目标网络与在线网络结构系统相同，参数和在线网络保持周期性一致。

算法出处
----------------------

**该算法的编写参考以下文献**:

`Reducing overestimation bias in multi-agent domains using double centralized critics 
<https://arxiv.org/pdf/1910.01465.pdf>`_

**论文引用信息**:

::

    @article{ackermann2019reducing,
        title={Reducing overestimation bias in multi-agent domains using double centralized critics},
        author={Ackermann, Johannes and Gabler, Volker and Osa, Takayuki and Sugiyama, Masashi},
        journal={arXiv preprint arXiv:1910.01465},
        year={2019}
    }