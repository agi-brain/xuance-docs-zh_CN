通用基础模块
======================

为了方便开发人员扩展模块功能，节约开发成本，玄策定义了六个主要的通用基础模块，
即数学工具、基础算子、日志工具、经验池、配置工具、统计工具。
这些基础模块在不同的算法中共享，具有很好的兼容性。下面针对这五个模块分别加以介绍。

数学工具
----------------------

算法库中需要用到的数学工具和基础算子定义在“./common/common_tools.py”文件中，
目前包含一个用于计算累积折扣回报的滤波器函数disount_cumsum()。
该函数输入为一个回合的奖励值序列和折扣因子 :math:`\gamma` ，输出为每一时刻的累计回报序列。
用户可在该模块中根据自己开发需要，增加其它数学工具。

基础算子
----------------------

基础算子主要涉及对神经网络权重参数和梯度相关的操作，
包含get_flat_grad，get_flat_params，assign_from_flat_grads，assign_from_flat_params。
这些基础算子分别在“./xuance_torch/utils/operators.py”和“./xuance_mindspore/utils/operators.py”中实现定义，
分别用于实现PyTorch和MindSpore两种框架下的强化学习算法设计。用户可根据所设计算法的需求，在该文件中扩展基础算子。

日志工具
----------------------
玄策利用tensorboard.writer模块中的SummaryWriter类对算法训练过程中产生的中间结果进行存储，并实现训练数据的可视化。

经验回放池
----------------------
经验池是强化学习算法的重要组成部分，主要存储智能体同环境交互的原始数据以及参与模型训练的其它数据。
根据在轨策略和离轨策略的不同，经验池分为基于on policy和基于off policy两大类。
单智能体DRL和MARL的经验池类定义分别位于./common/memory_tools.py和./common/memory_tools_marl.py两个文件中。
具体用法请参加算法设计部分。

参数配置工具
----------------------
针对不同算法的参数配置，该软件采用.yaml文件格式进行设置和存储，并利用yaml工具对配置文件进行读取。
同时结合sys.argv工具对命令行参数进行解析，并整合到最终的算法的参数字典变量中。

统计工具
----------------------
强化学习算法所需要的概率统计学工具在./common/statistic_tools.py及./xuance_torch/utils/distributions.py文件中定义，
其中包括数据归一化工具类RunningMeanStd，动作噪声类OUNoise，概率分布基类Distribution等。
