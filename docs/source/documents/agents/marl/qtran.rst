QTRAN
======================

算法描述
----------------------

QTRAN算法的全称是Q-transformation，该算法认为VDN的加和约束严重限制了值函数的函数表达能力，无法解决非单调协作任务。
QTRAN算法指出，更加通用的方法应该要满足独立-全局最大化（Independent-Global-Max, IGM）条件，
为了满足该条件，QTRAN算法将原始的整体值函数映射至新的值函数，使得它们在IGM条件下是等价的。

**论文链接**: `Qtran: Learning to factorize with transformation for cooperative multi-agent reinforcement learning 
<http://proceedings.mlr.press/v97/son19a/son19a.pdf>`_

**引用信息**:

::

    @inproceedings{son2019qtran,
        title={Qtran: Learning to factorize with transformation for cooperative multi-agent reinforcement learning},
        author={Son, Kyunghwan and Kim, Daewoo and Kang, Wan Ju and Hostallero, David Earl and Yi, Yung},
        booktitle={International conference on machine learning},
        pages={5887--5896},
        year={2019},
        organization={PMLR}
    }
