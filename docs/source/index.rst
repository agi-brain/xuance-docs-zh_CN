.. XuanCe documentation master file, created by
   sphinx-quickstart on Wed May 31 20:18:19 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎来到“玄策”中文文档!
======================================

.. raw:: html

   <a href="https://pypi.org/project/xuance/">
        <img alt="pypi" src="https://img.shields.io/pypi/v/xuance">
   </a>
   <a href="https://xuance.readthedocs.io">
        <img alt="pypi" src="https://readthedocs.org/projects/xuance/badge/?version=latest">
   </a>
   <a href="https://github.com/agi-brain/xuance/blob/master/LICENSE.txt">
        <img alt="pypi" src="https://img.shields.io/github/license/agi-brain/xuance">
   </a>
   <a href="https://pepy.tech/project/xuance">
        <img alt="pypi" src="https://static.pepy.tech/badge/xuance">
   </a>
   <a href="https://github.com/agi-brain/xuance/stargazers">
        <img alt="pypi" src="https://img.shields.io/github/stars/agi-brain/xuance?style=social">
   </a>
   <a href="https://github.com/agi-brain/xuance/forks">
        <img alt="pypi" src="https://img.shields.io/github/forks/agi-brain/xuance?style=social">
   </a>
   <a href="https://github.com/agi-brain/xuance/watchers">
        <img alt="pypi" src="https://img.shields.io/github/watchers/agi-brain/xuance?style=social">
   </a>

   <a href="https://pytorch.org/get-started/locally/">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%3E%3D1.13.0-red">
   </a>
   <a href="https://www.tensorflow.org/install">
        <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-%3E%3D2.6.0-orange">
   </a>
   <a href="https://www.mindspore.cn/install/en">
        <img alt="MindSpore" src="https://img.shields.io/badge/MindSpore-%3E%3D1.10.1-blue">
   </a>

   <a href="https://www.gymlibrary.dev/">
        <img alt="gymnasium" src="https://img.shields.io/badge/gymnasium-%3E%3D0.28.1-blue">
   </a>
   <a href="https://pettingzoo.farama.org/">
        <img alt="pettingzoo" src="https://img.shields.io/badge/PettingZoo-%3E%3D1.23.0-blue">
   </a>
   <a href="https://img.shields.io/pypi/pyversions/xuance">
        <img alt="Python" src="https://img.shields.io/pypi/pyversions/xuance">
   </a>

.. raw:: html

   <br>

**玄策** 是一个深度强化学习（Deep Reinforcement Learning, DRL）开源算法库。“玄”字意为玄妙的，“策”意为策略。
在深度强化学习算法中，智能体通过和环境交互不断试错，最终学习出一个最优策略完成任务，而不需要对环境或动力学模型建立精确的模型，因此该算法库被称为“玄妙的策略”，故而取名“玄策”。

此外，虽然深度强化学习能够解决很多复杂的任务，但是在算法调试的过程中，深度神经网络和优化过程对超参数往往比较敏感。
对于某特殊结构的算法，要想调出一组最佳的超参数，往往需要开发人员进行大量的试错。
由于对超参数调试的方法主要以来开发人员的经验，难以总结出一条通用的规律，因此常被戏称为“玄学”。
而该算法库提供了大量目前主流的DRL算法，其实现过程易于理解，使得算法的复现不再玄学。

玄策强化学习算法库目前同时支持多种深度学习框架，包括
PyTorch_ (|_1| |torch| |_1|)，
TensorFlow_ (|_1| |tensorflow| |_1|)，和
MindSpore_ (|_1| |mindspore| |_1|)。并且支持CPU、GPU、Ascend运算，能够在Linux，Windows，MacOS等操作系统上运行。

.. _PyTorch: https://pytorch.org/
.. _TensorFlow: https://www.tensorflow.org/
.. _MindSpore: https://www.mindspore.cn/en

| **GitHub**: `https://github.com/agi-brain/xuance.git <https://github.com/agi-brain/xuance.git/>`_

为什么选择“玄策”？
-----------------------

XuanCe 旨在简化深度强化学习算法的实现与开发流程，帮助研究者快速掌握核心原理，从而高效投入算法设计与创新。
其主要特性如下：

- **高度模块化**：采用模块化架构设计，具备优异的灵活性与可扩展性。
- **易学易用**：上手简单，安装便捷，适合不同层次的用户使用。
- **灵活的模型集成**：支持模型的自由组合与自定义配置，满足多样化需求。
- **丰富的算法库**：内置多种强化学习算法，覆盖多类型任务场景。
- **多任务场景支持**：同时支持深度强化学习（DRL）与多智能体强化学习（MARL）任务。
- **广泛的兼容性**：兼容 PyTorch、TensorFlow、MindSpore 等框架，并可高效运行于 CPU、GPU 以及 Linux、Windows、macOS 等平台。
- **高性能计算**：基于向量化环境实现快速执行与高效训练。
- **分布式训练**：支持多 GPU 并行训练，便于大规模实验扩展。
- **自动化超参数调优**：内置超参数自动搜索与优化功能。
- **可视化增强**：集成 TensorBoard 与 Weights & Biases（wandb）等工具，提供直观、全面的训练过程可视化。

“玄策”算法列表
--------------------------

**Value-based:**

* :class:`DQN` : :doc:`Deep Q-Network (DQN) <documents/algorithms/drl/dqn>`.
* :class:`DDQN` : :doc:`Double Deep Q-Network (Double DQN) <documents/algorithms/drl/ddqn>`.
* :class:`DuelDQN` : :doc:`Dueling Deep Q-Network (Dueling DQN) <documents/algorithms/drl/dueldqn>`.
* :class:`PerDQN` : :doc:`DQN with Prioritized Experience Replay (PER DQN) <documents/algorithms/drl/perdqn>`.
* :class:`NoisyDQN` : :doc:`DQN with Noisy Layers (Noisy DQN) <documents/algorithms/drl/noisydqn>`.
* :class:`DRQN` : :doc:`Deep Recurrent Q-Network (DRQN) <documents/algorithms/drl/drqn>`.
* :class:`QRDQN` : :doc:`DQN with Quantile Regression (QR-DQN) <documents/algorithms/drl/qrdqn>`.
* :class:`C51` : :doc:`Categorical 51 DQN (C51) <documents/algorithms/drl/c51>`.

**Policy-based:**

* :class:`PG` : :doc:`Policy Gradient (PG) <documents/algorithms/drl/pg>`.
* :class:`NPG` : :doc:`Natural Policy Gradient (NPG) <documents/algorithms/drl/npg>`.
* :class:`PPG` : :doc:`Phasic Policy Gradient (PPG) <documents/algorithms/drl/ppg>`.
* :class:`A2C` : :doc:`Advantage Actor Critic (A2C) <documents/algorithms/drl/a2c>`.
* :class:`SAC` : :doc:`Soft Actor-Critic (SAC) <documents/algorithms/drl/sac>`.
* :class:`PPOCLIP` : :doc:`Proximal Policy Optimization with Clipped Objective (PPO-Clip) <documents/algorithms/drl/ppoclip>`.
* :class:`PPOKL` : :doc:`Proximal Policy Optimization with KL Divergence (PPO-KL) <documents/algorithms/drl/ppokl>`.
* :class:`DDPG` : :doc:`Deep Deterministic Policy Gradient (DDPG) <documents/algorithms/drl/ddpg>`.
* :class:`TD3` : :doc:`Twin Delayed Deep Deterministic Policy Gradient (TD3) <documents/algorithms/drl/td3>`.
* :class:`PDQN` : :doc:`Parameterised Deep Q-Network (P-DQN) <documents/algorithms/drl/pdqn>`.
* :class:`MPDQN` : :doc:`Multi-pass Parameterised Deep Q-Network (MP-DQN) <documents/algorithms/drl/mpdqn>`.
* :class:`SPDQN` : :doc:`Split parameterised Deep Q-Network (SP-DQN) <documents/algorithms/drl/spdqn>`.

**MARL-based:**

* :class:`IQL` : :doc:`Independent Q-Learning (IQL) <documents/algorithms/marl/iql>`.
* :class:`VDN` : :doc:`Value Decomposition Networks (VDN) <documents/algorithms/marl/vdn>`.
* :class:`QMIX` : :doc:`Q-Mixing Networks (QMIX) <documents/algorithms/marl/qmix>`.
* :class:`WQMIX` : :doc:`Weighted Q-Mixing Networks (WQMIX) <documents/algorithms/marl/wqmix>`.
* :class:`QTRAN` : :doc:`Q-Transformation (QTRAN) <documents/algorithms/marl/qtran>`.
* :class:`DCG` : :doc:`Deep Coordination Graphs (DCG) <documents/algorithms/marl/dcg>`.
* :class:`IDDPG` : :doc:`Independent Deep Deterministic Policy Gradient (IDDPG) <documents/algorithms/marl/iddpg>`.
* :class:`MADDPG` : :doc:`Multi-agent Deep Deterministic Policy Gradient (MADDPG) <documents/algorithms/marl/maddpg>`.
* :class:`IAC` : :doc:`Independent Actor-Critic (IAC) <documents/algorithms/marl/iac>`.
* :class:`COMA` : :doc:`Counterfactual Multi-agent Policy Gradient (COMA) <documents/algorithms/marl/coma>`.
* :class:`VDAC` : :doc:`Value-Decomposition Actor-Critic (VDAC) <documents/algorithms/marl/vdac>`.
* :class:`IPPO` : :doc:`Independent Proximal Policy Optimization (IPPO) <documents/algorithms/marl/ippo>`.
* :class:`MAPPO` : :doc:`Multi-agent Proximal Policy Optimization (MAPPO) <documents/algorithms/marl/mappo>`.
* :class:`MFQ` : :doc:`Mean-Field Q-Learning (MFQ) <documents/algorithms/marl/mfq>`.
* :class:`MFAC` : :doc:`Mean-Field Actor-Critic (MFAC) <documents/algorithms/marl/mfac>`.
* :class:`ISAC` : :doc:`Independent Soft Actor-Critic (ISAC) <documents/algorithms/marl/isac>`.
* :class:`MASAC` : :doc:`Multi-agent Soft Actor-Critic (MASAC) <documents/algorithms/marl/masac>`.
* :class:`MATD3` : :doc:`Multi-agent Twin Delayed Deep Deterministic Policy Gradient (MATD3) <documents/algorithms/marl/matd3>`.
* :class:`IC3Net` : :doc:`Individual Controlled Continuous Communication Model (IC3Net) <documents/algorithms/marl/ic3net>`.

**Model-based:**

* :class:`DreamerV2` : :doc:`Dreamer V2 <documents/algorithms/mbrl/dreamer_v2>`.
* :class:`DreamerV3` : :doc:`Dreamer V3 <documents/algorithms/mbrl/dreamer_v3>`.
* :class:`HarmonyDreamer` : :doc:`HarmonyDreamer <documents/algorithms/mbrl/harmony_dream>`.

**Contrastive RL:**

* :class:`CURL` : :doc:`Contrastive Unsupervised Representations for Reinforcement Learning (CURL) <documents/algorithms/crl/curl>`.
* :class:`DrQ` : :doc:`Data-Regularized Q-Learning (DrQ) <documents/algorithms/crl/drq>`.
* :class:`SPR` : :doc:`Self-Predictive Representations for Reinforcement Learning (SPR) <documents/algorithms/crl/spr>`.

**Offline RL:**

* :class:`TD3BC` : :doc:`Twin Delayed Deep Deterministic Policy Gradient with Behavior Cloning (TD3BC) <documents/algorithms/offline/td3bc>`.

“玄策”整体框架
------------------------------------------

“玄策”的整体框架如下图所示.

.. image:: _static/figures/xuance_framework.png


玄策框架主要由以下四个部分构成:

- 第一部分: Configs. 环境参数、算法超参数、模型规模、训练参数等配置信息；
- 第二部分: Common Tools. 通用工具，包含经验回放池等模块；
- 第三部分: Environments. 环境模块，包含玄策的环境封装，向量化环境等工具；
- 第四部分: Algorithms. 算法模块，包含表征器、策略、学习器、智能体等模块。

“玄策”适用人群
------------------------------------------

“玄策”的适用人群包括但不限于：

- **研究人员**：深度强化学习方向的研究人员
- **开发人员**：深度强化学习算法开发人员
- **学生、初学者**：深度强化学习方向的学生、入门该方向的初学者
- **AI从业者**：从事 AI 行业，特别是对 AI 决策领域感兴趣的从业者

.. raw:: html

   <br><hr>

文档目录
------------------------------------------

.. toctree::
   :maxdepth: 2
   :caption: 教程

   documents/usage/installation
   documents/usage/basic_usage
   documents/usage/further_usage
   documents/usage/custom_env
   documents/usage/custom_algorithm
   documents/usage/custom_callback

.. toctree::
   :maxdepth: 2
   :caption: 算法:

   单智能体强化学习 <documents/algorithms/drl>
   多智能体强化学习 <documents/algorithms/marl>
   基于模型强化学习 <documents/algorithms/model_based_rl>
   对比强化学习 <documents/algorithms/crl>
   离线强化学习 <documents/algorithms/offline_rl>

.. toctree::
   :maxdepth: 5
   :caption: 接口：

   documents/api/common
   documents/api/configs
   documents/api/environments
   documents/api/torch
   documents/api/tensorflow
   documents/api/mindspore

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
   documents/CONTRIBUTING
   Contribute to the Docs (EN) <https://github.com/agi-brain/xuance/tree/master/docs>
   Contribute to the Docs (CN) <https://github.com/agi-brain/xuance-docs-zh_CN/tree/master/docs>
