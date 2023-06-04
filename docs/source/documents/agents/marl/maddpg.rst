MADDPG
======================

算法描述
----------------------
MADDPG算法全称为Multi-agent Deep Deterministic Policy Gradient，是一种基于Actor-Critic的多智能体强化学习算法。
MADDPG由IDDPG算法发展而来，其基本结构和网络的更新原理和IDDPG相同，
不同之处在于该算法中的Critic网络将所有智能体的局部观测、动作、OneHot标识作为输入并输出Q值，
通过获取全局的观测和动作信息实现算法性能的提升。

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