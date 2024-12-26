Convolutional Neural Networks
------------------------------

卷积神经网络（Convolutional Neural Networks, CNN）主要用于处理图像输入数据，提取出特征向量，一般输入类型为多通道图像矩阵，输出多维向量，
名称为 cnn_block,其定义位于./xuance_torch/utils/layers.py和./xuance_ms/utils/layers.py中。
实例化该类需要指定输入尺寸（input_shape），滤波方法（filter），核大小（kernel_size），步长（stride），
归一化方法（normalize），激活函数（activation），初始化方法（initialize）。
在pytorch下实现还需指定设备类型（device），以确定模型在CPU上运行还是GPU。


PyTorch
^^^^^^^^^^^^^

.. automodule:: xuance.torch.representations.cnn
    :members:
    :undoc-members:
    :show-inheritance:

TensorFlow2
^^^^^^^^^^^^^

.. automodule:: xuance.tensorflow.representations.cnn
    :members:
    :undoc-members:
    :show-inheritance:

MindSpore
^^^^^^^^^^^^^

.. automodule:: xuance.mindspore.representations.cnn
    :members:
    :undoc-members:
    :show-inheritance:
