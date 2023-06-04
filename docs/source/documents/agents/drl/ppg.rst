PPG
======================

算法描述
----------------------

PPG（Phasic Policy Gradient）是一种基于策略梯度的深度强化学习算法，
用于进一步通过加入辅助任务来提升强化学习的数据效率。
具体来说，由于在这类算法中，为了降低策略更新的期望偏差，
策略网络往往在更新次数较少的时候收益较大，而实际应用的时候，
策略网络往往会跟网络中的其它部分共享一些参数，那么为了对表征网络建立更高效的学习，
这一部分应该进行更多次的更新。
因此，在PPG算法中，网络的优化目标被划分成了三个部分，其一是策略梯度损失，其二是辅助任务损失，其三是值函数损失。
由于辅助任务与策略部分共享一部分参数，因此，在进行辅助任务损失更新的时候，加入了KL散度的限制保证更新前后策略差别不会太大。
然后通过调整策略更新和其它网络部分更新的频率，取长补短，最终提升了数据效率。

算法出处
----------------------

**论文链接**:
`Phasic policy gradient 
<http://proceedings.mlr.press/v139/cobbe21a/cobbe21a.pdf>`_

**论文引用信息**:

::

    @inproceedings{cobbe2021phasic,
        title={Phasic policy gradient},
        author={Cobbe, Karl W and Hilton, Jacob and Klimov, Oleg and Schulman, John},
        booktitle={International Conference on Machine Learning},
        pages={2020--2027},
        year={2021},
        organization={PMLR}
    }
