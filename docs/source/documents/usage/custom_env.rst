自定义环境
=============================

XuanCe 支持用户在使用内置环境的同时，创建并运行 **自定义环境（Custom Environments）**。
通过这一功能，用户可以将自己开发的环境与 XuanCe 内置的强化学习算法无缝结合。

在 XuanCe 中，你可以：

- 创建基于马尔可夫决策过程（MDP）的 **单智能体环境（Single-Agent Environment）**；
- 构建基于部分可观测马尔可夫决策过程（POMDP）的 **多智能体环境（Multi-Agent Environment）**；
- 使用 XuanCe 提供的算法（如 DQN、PPO、IPPO 等）直接运行这些自定义环境。

这一机制为研究者与开发者提供了极大的灵活性，
无论是机器人控制、金融交易、还是多智能体交互任务，都可以轻松集成和测试。

参阅以下小节以开始创建：

- :doc:`自定义环境: 单智能体 <custom_env/custom_drl_env>`
- :doc:`自定义环境: 多智能体 <custom_env/custom_marl_env>`

.. toctree::
   :hidden:
   :maxdepth: 1

   单智能体 <custom_env/custom_drl_env>
   多智能体 <custom_env/custom_marl_env>