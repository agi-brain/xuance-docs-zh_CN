Double DQN
======================

算法描述
----------------------

Double DQN (Deep reinforcement learning with double q-learning) 算法是对原始DQN算法的一种改进，
用于解决DQN算法在估计动作价值时可能出现的过高估计问题。
Double DQN的主要思想是将动作选择和动作评估分离，使用两个不同的Q网络来分别处理这两个过程，降低过高估计对学习过程的影响。
在Double DQN算法中，有两个Q网络，一个是估计Q网络（Evaluation Q-network，记为Q_e），
另一个是目标Q网络（Target Q-network，记为Q_t）。这两个网络的结构相同，但参数更新的方式有所不同。
在每一步更新中，智能体会利用估计Q网络Q_e来选择最优动作，然后用目标Q网络Q_t来评估该动作的价值。
这样可以减小单一Q网络在选择和评估过程中可能产生的过高估计偏差。

算法出处
----------------------

论文链接: `Deep reinforcement learning with double q-learning 
<https://ojs.aaai.org/index.php/AAAI/article/view/10295>`_

::

    @inproceedings{van2016deep,
        title={Deep reinforcement learning with double q-learning},
        author={Van Hasselt, Hado and Guez, Arthur and Silver, David},
        booktitle={Proceedings of the AAAI conference on artificial intelligence},
        volume={30},
        number={1},
        year={2016}
    }
