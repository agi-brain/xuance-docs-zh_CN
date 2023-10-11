DCG
======================

算法描述
----------------------

DCG算法全称为Deep Coordination Graph，是一种基于值函数的多智能体强化学习算法。
DCG将每个智能体看作一个节点，智能体之间的协作收益看作边，从而构成一个无向协作拓扑图。
DCG在每次决策过程中，通过信念传播算法获得稳定的联合动作作为多智能体系统的决策。
在优化过程中，系统的整体Q值由独立Q值和智能体两两联合收益构成，采用端到端训练。

算法出处
----------------------

**论文链接**:
`Deep coordination graphs 
<http://proceedings.mlr.press/v119/boehmer20a/boehmer20a.pdf>`_

**论文引用信息**:

::

    @inproceedings{bohmer2020deep,
        title={Deep coordination graphs},
        author={B{\"o}hmer, Wendelin and Kurin, Vitaly and Whiteson, Shimon},
        booktitle={International Conference on Machine Learning},
        pages={980--991},
        year={2020},
        organization={PMLR}
    }
