.. XuanCe documentation master file, created by
   sphinx-quickstart on Wed May 31 20:18:19 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎来到‘玄策’中文文档!
======================================

**玄策** 是一个深度强化学习（Deep Reinforcement Learning, DRL）开源算法库。“玄”字意为玄妙的，“策”意为策略。
在深度强化学习算法中，智能体通过和环境交互不断试错，最终学习出一个最优策略完成任务，而不需要对环境或动力学模型建立精确的模型，因此该算法库被称为“玄妙的策略”，故而取名“玄策”。

此外，虽然深度强化学习能够解决很多复杂的任务，但是在算法调试的过程中，深度神经网络和优化过程对超参数往往比较敏感。
对于某特殊结构的算法，要想调出一组最佳的超参数，往往需要开发人员进行大量的试错。
由于对超参数调试的方法主要以来开发人员的经验，难以总结出一条通用的规律，因此常被戏称为“玄学”。
而该算法库提供了大量目前主流的DRL算法，其实现过程易于理解，使得算法的复现不再玄学。
“玄策”的适用人群包括但不限于：人工智能方向学生，DRL开发人员，DRL入门学习者，等。

“玄策”目前同时支持多种深度学习框架，包括
PyTorch_，
TensorFlow_，和
MindSpore_。并且支持CPU、GPU运算，能够在Linux，Windows，MacOS等操作系统上运行。

.. _PyTorch: https://pytorch.org/
.. _TensorFlow: https://www.tensorflow.org/
.. _MindSpore: https://www.mindspore.cn/en

“玄策”的 **主要特征** 总结如下：
   * 支持PyTorch，TensorFlow2，MindSpore三种深度学习框架。
   * 可在Linux，Windows，MacOS等操作系统上运行。
   * 安装方便，代码易读，上手简单。
   * 支持丰富的算法，包括单智能体、多智能体协作、多智能体对抗博弈等任务。

目前，“玄策”已在GitHub和OpenI社区开源，链接如下：

| **GitHub**： `https://github.com/agi-brain/xuance.git <https://github.com/agi-brain/xuance.git/>`_
| **OpenI**： `https://github.com/agi-brain/xuance.git <https://github.com/agi-brain/xuance.git/>`_

The Framework of XuanCe
------------------------------------------

The overall framework of XuanCe is shown as below.

.. image:: _static/figures/xuance_framework.png


玄策框架主要由以下四个部分构成:

- 第一部分: Configs. 环境参数、算法超参数、模型规模、训练参数等配置信息；
- 第二部分: Common Tools. 通用工具，包含经验回放池等模块；
- 第三部分: Environments. 环境模块，包含玄策的环境封装，向量化环境等工具；
- 第四部分: Algorithms. 算法模块，包含表征器、策略、学习器、智能体等模块。

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: 教程

   documents/usage/installation
   documents/usage/basic_usage
   documents/usage/further_usage
   documents/usage/new_envs
   documents/usage/new_algorithm

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: 接口：

   documents/api/environments
   documents/api/agents
   documents/api/runners
   documents/api/representations
   documents/api/policies
   documents/api/learners
   documents/api/configs
   documents/api/common
   documents/api/utils

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: 参考基准

   documents/benchmark/mujoco
   documents/benchmark/atari
   documents/benchmark/smac

.. toctree::
   :hidden:
   :caption: 玄策开发

   Github <https://github.com/agi-brain/xuance.git>
   documents/release_log
   Contribute to XuanCe <https://github.com/agi-brain/xuance/pulls>
   Contribute to the Docs (English) <https://github.com/agi-brain/xuance/tree/master/docs>
   Contribute to the Docs (Chinese) <https://github.com/agi-brain/xuance-docs-zh_CN/tree/master/docs>
