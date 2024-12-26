Multi-Layer Perceptron
---------------------------

多层感知器（Multi-Layer Perceptron, MLP）是一种最简单的深层神经网络模型，用于处理向量输入，
用户可根据各自需要实例化多层感知器模块，
其定义位于./xuance_torch/utils/layers.py和./xuance_ms/utils/layers.py文件中，类名称为mlp_block。
实例化该类需指定输入维度大小（input_dim），输出维度大小（output_dim），归一化方法（normalize），
激活函数选择（activation），初始化方法（initialize）。
在pytorch下实现还需指定设备类型（device），以确定模型在CPU上运行还是GPU。

PyTorch
^^^^^^^^^^^^^

.. automodule:: xuance.torch.representations.mlp
    :members:
    :undoc-members:
    :show-inheritance:

TensorFlow2
^^^^^^^^^^^^^

.. automodule:: xuance.tensorflow.representations.mlp
    :members:
    :undoc-members:
    :show-inheritance:

MindSpore
^^^^^^^^^^^^^

.. automodule:: xuance.mindspore.representations.mlp
    :members:
    :undoc-members:
    :show-inheritance:
