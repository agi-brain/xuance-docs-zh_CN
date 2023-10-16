Policies
======================

离散动作随机策略
----------------------
针对离散动作随机策略，本软件根据pytorch和mindspore两种编程框架的差异，
通过不同的方式加以实现，相关定义均位于各自目录的policy子文件夹下。
其中，使用torch.distributions和mindspore.nn.probability.distribution类
分别实现pytorch和mindspore两种编程框架下的Categorical分布。
根据动作的维度大小确定Categorical的类别数量。
相关的策略包括：ActorCriticPolicy，ActorPolicy两种。

高斯分布随机策略
----------------------
针对连续动作随机策略，本软件根据pytorch和mindspore两种编程框架的差异，
通过不同的方式加以实现，相关定义均位于各自目录的policy子文件夹下。
其中，使用torch.distributions和mindspore.nn.probability.distribution工具
分别实现pytorch和mindspore两种编程框架下的Gaussian分布。
根据动作的维度大小确定均值和方差的维度。相关的策略包括：ActorCriticPolicy，ActorPolicy两种。

确定性策略
----------------------
确定性策略主要用于DQN和DDPG两大类算法的策略实现。
确定性策略包括BasicQnetwork, DuelQnetwork, DDPGPolicy, TD3Policy, 
MixingQnetwork, Weighted_MixingQnetwork, Qtran_MixingQnetwork, 
DCG_policy,Basic_DDPG_policy，MADDPG_policy等。
相关定义均位于./policy子文件下。
确定性策略不包含概率分布，因此不需要用到distribution工具。
其包含的Q网络、Actor网络和Critic网络，均建立在状态表征网络基础上，利用2.3节中介绍的多层感知器模块实现。

