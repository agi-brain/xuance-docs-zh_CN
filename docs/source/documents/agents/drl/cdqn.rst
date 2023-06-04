Convolutional DQN
======================

算法描述
----------------------

CDQN：CDQN全称为Convolutional Neural Network-based Deep Q-Network，
是一种基于值(value-based)的深度强化学习算法。
CDQN使用神经网络来逼近行为值函数，使用了target Q network来更新target，并使用经验回放Experience replay。
CDQN将CNN算法和DQN相结合，通过卷积神经网络对网络的输入进行处理，其中卷积层主要作用是提取特征，
池化层主要作用是下采样，却不会损坏识别结果，全连接层主要作用是分类。
CNN算法在图像和语音识别方面的优势很大，再使用DQN算法对整个网络进行训练，
输出使Q值最大的动作给环境，得到新的状态。

算法出处
----------------------

论文链接: `Convolutional neural network-based deep Q-network (CNN-DQN) resource management in cloud radio access network
<https://ieeexplore.ieee.org/abstract/document/9867958/>`_

论文引用信息:

::

    @article{iqbal2022convolutional,
        title={Convolutional neural network-based deep Q-network (CNN-DQN) resource management in cloud radio access network},
        author={Iqbal, Amjad and Tham, Mau-Luen and Chang, Yoong Choon},
        journal={China Communications},
        volume={19},
        number={10},
        pages={129--142},
        year={2022},
        publisher={IEEE}
    }
