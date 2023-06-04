CLDQN
======================

算法描述
----------------------

CLDQN全称为Convolutional Neural Network and Long Short Term Memory-based Deep Q-Network，
是一种基于值(value-based)的深度强化学习算法。CLDQN使用神经网络来逼近行为值函数，
使用了target Q network来更新target，并使用经验回放Experience replay。
CLDQN将CNN-LSTM（时空网络）算法和DQN相结合，首先利用卷积神经网络将输入提取出有用的特征，
再通过长短时记忆网络，利用时间序列对输入进行分析，可以有效的传递和表达长时间序列中的信息并且不会导致长时间前的有用信息被忽略，
再使用DQN算法对整个网络进行训练，输出使Q值最大的动作给环境，得到新的状态。

算法出处
----------------------

论文链接: `UAV target following in complex occluded environments with adaptive multi-modal fusion 
<https://link.springer.com/article/10.1007/s10489-022-04317-2>`_

论文引用信息:

::

    @article{xu2022uav,
        title={UAV target following in complex occluded environments with adaptive multi-modal fusion},
        author={Xu, Lele and Wang, Teng and Cai, Wenzhe and Sun, Changyin},
        journal={Applied Intelligence},
        pages={1--17},
        year={2022},
        publisher={Springer}
    }
