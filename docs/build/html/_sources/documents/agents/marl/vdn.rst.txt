VDN
======================

算法描述
----------------------

VDN算法全称为value decomposition networks，是一种基于值函数的多智能体强化学习算法，
其网络结构和IQL相同，不同之处在于它的网络参数更新方式。
VDN采用加和的方式将各智能体的值函数经过VDN mixer模块整合为一个值函数，并将其作为优化目标，从而实现端到端训练。
这种方法能够有效缓解环境不稳定性问题和局部观测问题，被广泛使用。

算法出处
----------------------

**论文链接**: `Value-decomposition networks for cooperative multi-agent learning 
<https://arxiv.org/pdf/1706.05296>`_

**引用信息**:

::

    @article{sunehag2017value,
        title={Value-decomposition networks for cooperative multi-agent learning},
        author={Sunehag, Peter and Lever, Guy and Gruslys, Audrunas and Czarnecki, Wojciech Marian and Zambaldi, Vinicius and Jaderberg, Max and Lanctot, Marc and Sonnerat, Nicolas and Leibo, Joel Z and Tuyls, Karl and others},
        journal={arXiv preprint arXiv:1706.05296},
        year={2017}
    }
