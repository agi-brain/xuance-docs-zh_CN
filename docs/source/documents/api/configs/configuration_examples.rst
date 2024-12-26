Configuration Examples
------------------------------------------------------

以DQN算法在Atari环境中的参数配置为例，除了基础参数配置外，其算法参数配置存放于 xuance/configs/dqn/atari.yaml
文件中。

由于Atari环境中一共超过60个不同场景，场景比较统一，只是任务不同，因此只需要一个默认的参数配置文件即可。

针对场景差异较大的环境，如 ``Box2D`` 环境中的 ``CarRacing-v2`` 和 ``LunarLander`` 场景，
前者的状态输入是96*96*3的RGB图像，后者则是一个8维向量。因此，针对这两个场景的DQN算法参数配置分别存于以下两个文件中：

    * xuance/configs/dqn/box2d/CarRacing-v2.yaml
    * xuance/configs/dqn/box2d/LunarLander-v2.yaml

Within the following content, we provid the preset arguments for each implementation that can be run by following the steps in :doc:`Quick Start </documents/usage/basic_usage>`.

.. include:: example_value_based.rst
.. include:: example_policy_based.rst
.. include:: example_marl.rst
