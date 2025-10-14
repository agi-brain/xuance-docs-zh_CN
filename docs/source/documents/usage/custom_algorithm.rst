自定义算法
=============================

XuanCe **提供了一个灵活的框架，使用户能够** 设计并实现自己的强化学习算法。
除了内置的大量算法之外，用户还可以利用 XuanCe 的模块化设计来开发新的方法。

XuanCe 中的所有核心算法组件——例如神经网络结构、经验回放缓存、优化器以及策略（policy）——都以可扩展、可复用的方式实现。
这意味着用户无需从零开始，只需通过定制或组合现有模块，就能轻松构建一个新的算法。

使用 XuanCe，您可以：

- 开发 **单智能体算法（Single-Agent Algorithms）**，基于标准强化学习框架，如 DQN、PPO、SAC、TD3 等。
- 构建 **多智能体算法（Multi-Agent Algorithms）**，扩展自 QMIX、MAPPO、MADDPG 等典型架构。
- 直接使用相同的训练与评估流程，将自定义算法与 XuanCe 的成熟基线（baseline）进行对比。

这种设计为科研与实验带来了显著优势：

- 您可以专注于算法的 **创新**，而无需重复实现基础的强化学习基础设施。
- 您可以 **复用** XuanCe 中丰富的基准算法集合作为**基线（baseline）**，从而节省大量时间与精力。
- 您可以 **测试与可视化** 实验结果，得益于统一的日志记录、回放与评估接口。

若要开始使用，请参阅以下内容：

- :doc:`自定义算法：DRL <custom_algorithm/custom_drl_algorithm>`
- :doc:`自定义算法：MARL <custom_algorithm/custom_marl_algorithm>`

.. toctree::
   :hidden:
   :maxdepth: 1

   DRL <custom_algorithm/custom_drl_algorithm>
   MARL <custom_algorithm/custom_marl_algorithm>
