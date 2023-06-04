QMIX
======================

算法描述
----------------------

QMIX算法全称为Q-Mixing networks，该算法同样是在VDN算法基础上发展而来。
由于VDN利用加和约束优化各智能体的值函数，极大地限制了算法在复杂问题上的决策能力。
为了解决该问题，QMIX算法将VDN的加和性约束放宽至单调性约束。
QMIX算法将各智能体的独立Q值经过qmixer模块，得到整体Q值，该整体Q值和独立Q值保持单调性关系。
QMIX的更新方式同VDN一样，使用端到端训练。

算法出处
----------------------

论文链接：

- `QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning 
<http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf>`_

- `Monotonic value function factorisation for deep multi-agent reinforcement learning 
<https://www.jmlr.org/papers/volume21/20-081/20-081.pdf>`_

引用信息：

::

    @InProceedings{pmlr-v80-rashid18a,
        title = {{QMIX}: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning},
        author = {Rashid, Tabish and Samvelyan, Mikayel and Schroeder, Christian and Farquhar, Gregory and Foerster, Jakob and Whiteson, Shimon},
        booktitle = {Proceedings of the 35th International Conference on Machine Learning},
        pages = {4295--4304},
        year = {2018},
        editor = {Dy, Jennifer and Krause, Andreas},
        volume = {80},
        series = {Proceedings of Machine Learning Research},
        month = {10--15 Jul},
        publisher = {PMLR},
        pdf = {http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf},
        url = {https://proceedings.mlr.press/v80/rashid18a.html},
        abstract = {In many real-world settings, a team of agents must coordinate their behaviour while acting in a decentralised way. At the same time, it is often possible to train the agents in a centralised fashion in a simulated or laboratory setting, where global state information is available and communication constraints are lifted. Learning joint action-values conditioned on extra state information is an attractive way to exploit centralised learning, but the best strategy for then extracting decentralised policies is unclear. Our solution is QMIX, a novel value-based method that can train decentralised policies in a centralised end-to-end fashion. QMIX employs a network that estimates joint action-values as a complex non-linear combination of per-agent values that condition only on local observations. We structurally enforce that the joint-action value is monotonic in the per-agent values, which allows tractable maximisation of the joint action-value in off-policy learning, and guarantees consistency between the centralised and decentralised policies. We evaluate QMIX on a challenging set of StarCraft II micromanagement tasks, and show that QMIX significantly outperforms existing value-based multi-agent reinforcement learning methods.}
    }


    @article{rashid2020monotonic,
        title={Monotonic value function factorisation for deep multi-agent reinforcement learning},
        author={Rashid, Tabish and Samvelyan, Mikayel and De Witt, Christian Schroeder and Farquhar, Gregory and Foerster, Jakob and Whiteson, Shimon},
        journal={The Journal of Machine Learning Research},
        volume={21},
        number={1},
        pages={7234--7284},
        year={2020},
        publisher={JMLRORG}
    }