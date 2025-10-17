Configuration Examples
------------------------------------------------------

以 Atari 环境中的 DQN 算法为例，除了基本参数配置外，与算法相关的特定参数还存储在 “xuance/configs/dqn/atari.yaml” 文件中。

由于 Atari 环境中包含 60 多种不同的场景，这些场景之间的差异主要体现在任务上而非环境结构上，因此使用一个默认的参数配置文件即可满足大多数情况的需求。

对于场景差异较大的环境（例如 “Box2D” 环境中的 “CarRacing-v2” 和 “LunarLander” 场景），前者的状态输入为大小为 96×96×3 的 RGB 图像，
而后者的状态输入则是一个 8 维向量。因此，针对这两种场景的 DQN 算法参数配置分别保存在以下两个文件中:

    * xuance/configs/dqn/box2d/CarRacing-v2.yaml
    * xuance/configs/dqn/box2d/LunarLander-v2.yaml

Within the following content, we provide the preset arguments for each implementation that can be run by following the steps in :doc:`Quick Start </documents/usage/basic_usage>`.
在接下来的内容中，我们将为每个实现提供预设参数，这些参数可以按照 :doc:`快速开始 </documents/usage/basic_usage>` 中的步骤直接运行。

.. include:: example_value_based.rst
.. include:: example_policy_based.rst
.. include:: example_marl.rst
