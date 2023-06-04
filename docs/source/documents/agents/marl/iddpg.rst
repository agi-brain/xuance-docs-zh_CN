IDDPG
======================

算法描述
----------------------
IDDPG算法全称为Independent Deep Deterministic Policy Gradient，是一种基于Actor-Critic的多智能体强化学习算法。
IDDPG由DDPG算法发展而来，其基本结构和网络的更新原理和DDPG相同，不同之处在于该算法为每个智能体配备Actor和Critic网络，
网络参数共享，Actor网络将该智能体的局部观测、OneHot标识作为输入并输出动作值，
Critic网络将该智能体的局部观测、动作、OneHot标识作为输入并输出Q值。

算法出处
----------------------

**该算法的编写参考以下文献**:

`Multi-agent actor-critic for mixed cooperative-competitive environments 
<https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf>`_

**论文引用信息**:

::

    @article{lowe2017multi,
        title={Multi-agent actor-critic for mixed cooperative-competitive environments},
        author={Lowe, Ryan and Wu, Yi I and Tamar, Aviv and Harb, Jean and Pieter Abbeel, OpenAI and Mordatch, Igor},
        journal={Advances in neural information processing systems},
        volume={30},
        year={2017}
    }
