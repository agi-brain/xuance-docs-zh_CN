状态表征
======================

多层感知器模块
----------------------

多层感知器是一种最简单的深层神经网络模型，用于处理向量输入，
用户可根据各自需要实例化多层感知器模块，
其定义位于./xuance_torch/utils/layers.py和./xuance_ms/utils/layers.py文件中，类名称为mlp_block。
实例化该类需指定输入维度大小（input_dim），输出维度大小（output_dim），归一化方法（normalize），
激活函数选择（activation），初始化方法（initialize）。
在pytorch下实现还需指定设备类型（device），以确定模型在CPU上运行还是GPU。

卷积神经网络模块
----------------------
卷积神经网络主要用于处理图像输入数据，提取出特征向量，一般输入类型为多通道图像矩阵，输出多维向量，
名称为 cnn_block,其定义位于./xuance_torch/utils/layers.py和./xuance_ms/utils/layers.py中。
实例化该类需要指定输入尺寸（input_shape），滤波方法（filter），核大小（kernel_size），步长（stride），
归一化方法（normalize），激活函数（activation），初始化方法（initialize）。
在pytorch下实现还需指定设备类型（device），以确定模型在CPU上运行还是GPU。

循环神经网络模块
----------------------
循环神经网络主要用于处理时序信号信息，提取出当前时序信号的特征向量。
根据使用场景差异，本软件提供两种循环神经网路模块：gru_block和lstm_block，
其定义均位于./xuance_torch/utils/layers.py和./xuance_ms/utils/layers.py中。
实例化该类需指定输入维度大小（input_dim），输出维度大小（output_dim），
剪枝方法（droupout），初始化方法（initialize）。同样地，在pytorch下实现还需指定设备类型（device），
以确定模型在CPU上运行还是GPU。