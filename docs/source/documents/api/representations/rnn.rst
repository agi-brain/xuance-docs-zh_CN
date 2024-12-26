Recurrent Neural Networks
------------------------------

循环神经网络（Recurrent Neural Networks, RNN）主要用于处理时序信号信息，提取出当前时序信号的特征向量。
根据使用场景差异，本软件提供两种循环神经网路模块：gru_block和lstm_block，
其定义均位于./xuance_torch/utils/layers.py和./xuance_ms/utils/layers.py中。
实例化该类需指定输入维度大小（input_dim），输出维度大小（output_dim），
剪枝方法（droupout），初始化方法（initialize）。同样地，在pytorch下实现还需指定设备类型（device），
以确定模型在CPU上运行还是GPU。

PyTorch
^^^^^^^^^^^^^

.. automodule:: xuance.torch.representations.rnn
    :members:
    :undoc-members:
    :show-inheritance:

TensorFlow2
^^^^^^^^^^^^^

.. automodule:: xuance.tensorflow.representations.rnn
    :members:
    :undoc-members:
    :show-inheritance:

MindSpore
^^^^^^^^^^^^^

.. automodule:: xuance.mindspore.representations.rnn
    :members:
    :undoc-members:
    :show-inheritance:
