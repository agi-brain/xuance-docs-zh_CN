IQL
======================

算法描述
----------------------

IQL算法全称为Independent Q-leanring, 是一种基于值函数的多智能体强化学习算法。
IQL采用独立学习的方法实现各智能体的决策，智能体的Q网络参数共享，并在输入端以One-Hot向量区分各智能体标识。
算法整体采用集中式训练和分布式执行的框架，实现端到端训练。
IQL是一种实现多智能体强化学习最直接的方式，虽然它存在环境不稳定、局部观测等问题，
但是其结构简单，容易在简单的多智能体协作问题中快速部署。

在本算法库中，IQL的网络分为representation和eval_Q两部分，其网络结构和DQN算法相似。
不同之处在于，IQL的representation模块输入信息中包含该智能体的局部观测和身份标识。

算法出处
----------------------

**参考论文链接**: `Independent reinforcement learners in cooperative markov games: a survey regarding coordination problems 
<https://hal.science/file/index/docid/720669/filename/Matignon2012independent.pdf>`_

**参考论文引用信息**:

::

    @article{matignon2012independent,
        title={Independent reinforcement learners in cooperative markov games: a survey regarding coordination problems},
        author={Matignon, Laetitia and Laurent, Guillaume J and Le Fort-Piat, Nadine},
        journal={The Knowledge Engineering Review},
        volume={27},
        number={1},
        pages={1--31},
        year={2012},
        publisher={Cambridge University Press}
    }
