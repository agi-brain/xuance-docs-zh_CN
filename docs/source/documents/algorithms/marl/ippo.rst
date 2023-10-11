MAPPO
======================

算法描述
----------------------

MAPPO算法全称为Multi-agent Proximal Policy Optimization
是从PPO算法发展而来的多智能体强化学习算法，它的基本结构和PPO算法类似，
不同之处在于其策略网络和值函数网络均加上One-Hot向量作为智能体身份标识，以区分各不同智能体的决策信息。
MAPPO的整体值函数采用加和的方式获取，同VDN类似。

算法出处
----------------------

**论文链接**:
`The surprising effectiveness of ppo in cooperative multi-agent games 
<https://proceedings.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf>`_

**论文引用信息**:

::

    @article{yu2022surprising,
        title={The surprising effectiveness of ppo in cooperative multi-agent games},
        author={Yu, Chao and Velu, Akash and Vinitsky, Eugene and Gao, Jiaxuan and Wang, Yu and Bayen, Alexandre and Wu, Yi},
        journal={Advances in Neural Information Processing Systems},
        volume={35},
        pages={24611--24624},
        year={2022}
    }
