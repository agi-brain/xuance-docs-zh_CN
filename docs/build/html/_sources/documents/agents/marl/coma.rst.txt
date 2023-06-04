COMA
======================

算法描述
----------------------
COMA算法全称为Counterfactual Multi-agent Policy Gradient，是一种基于策略的多智能体强化学习算法。
该算法提出利用反事实推断获得系统整体的优势函数，从而实现值函数的更新。
针对各智能体的更新，采用多个Actor网络作为参数化策略，并从值函数网络得到各智能体策略的梯度，从而缓解lazy agent问题。

算法出处
----------------------

论文链接：

- `Counterfactual multi-agent policy gradients 
<https://ojs.aaai.org/index.php/AAAI/article/view/11794>`_

论文引用信息：

::

    @inproceedings{foerster2018counterfactual,
        title={Counterfactual multi-agent policy gradients},
        author={Foerster, Jakob and Farquhar, Gregory and Afouras, Triantafyllos and Nardelli, Nantas and Whiteson, Shimon},
        booktitle={Proceedings of the AAAI conference on artificial intelligence},
        volume={32},
        number={1},
        year={2018}
    }
