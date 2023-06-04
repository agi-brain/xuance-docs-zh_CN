Noisy DQN
======================

算法描述
----------------------

NoisyNet DQN是对原始DQN算法的一种改进，旨在通过引入噪声来探索更多的状态和动作空间，从而提高强化学习性能。
这种方法的主要思想是在神经网络的参数中添加噪声，使得智能体在学习过程中自然地产生探索行为，而无需额外引入如ε-greedy等探索策略。
在NoisyNet DQN中，网络中的权重参数w和偏置参数b会被分别替换为包含噪声的参数。

算法出处
----------------------

论文链接: `Noisy networks for exploration 
<https://arxiv.org/pdf/1706.10295.pdf>`_

论文引用信息:

::

    @article{fortunato2017noisy,
        title={Noisy networks for exploration},
        author={Fortunato, Meire and Azar, Mohammad Gheshlaghi and Piot, Bilal and Menick, Jacob and Osband, Ian and Graves, Alex and Mnih, Vlad and Munos, Remi and Hassabis, Demis and Pietquin, Olivier and others},
        journal={arXiv preprint arXiv:1706.10295},
        year={2017}
    }
