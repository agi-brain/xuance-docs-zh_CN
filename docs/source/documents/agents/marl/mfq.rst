MFQ
======================

算法描述
----------------------
MFQ算法全称为Mean-Field Q-Leanring，是一种基于值函数的多智能体强化学习算法。
MFQ的核心思想是利用中心场的概念，将每个智能体同其邻居看作一个局部的决策场景，
各智能体的Q值函数输入包含其局部观测、邻居的上一时刻动作信息。
智能体的动作由各动作下的Q值通过bolzmann函数输出，作为该智能体的决策信息。

算法出处
----------------------

论文链接：

- `Mean field multi-agent reinforcement learning 
<http://proceedings.mlr.press/v80/yang18d/yang18d.pdf>`_

论文引用信息：

::

    @inproceedings{yang2018mean,
        title={Mean field multi-agent reinforcement learning},
        author={Yang, Yaodong and Luo, Rui and Li, Minne and Zhou, Ming and Zhang, Weinan and Wang, Jun},
        booktitle={International conference on machine learning},
        pages={5571--5580},
        year={2018},
        organization={PMLR}
    }
