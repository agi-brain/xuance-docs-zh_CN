学习器Learner类
======================

学习器功能介绍
----------------------
Learner类是每个DRL算法优化模型参数的关键部分，其初始化输入包含策略网络（policy），优化器（optimizer），
学习率衰减模块（scheduler），日志模块（summary_writer），模型存储地址（modeldir）等。
Learner类的主要功能是更新网络参数，在成员函数update()中加以实现，具体内容因各种算法的不同而异。
