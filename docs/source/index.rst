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
“玄策”的适用人群包括但不限于：人工智能方向学生，DRL开发人员，DRL入门学习者，等。

“玄策”目前同时支持多种深度学习框架，包括
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

* :class:`DQN_Agent` : :doc:`Deep Q-Network (DQN) <documents/api/agents/drl/dqn_agent>`.
* :class:`DDQN_Agent` : :doc:`Double Deep Q-Network (Double DQN) <documents/api/agents/drl/ddqn_agent>`.
* :class:`DuelDQN_Agent` : :doc:`Dueling Deep Q-Network (Dueling DQN) <documents/api/agents/drl/dueldqn_agent>`.
* :class:`PerDQN_Agent` : :doc:`DQN with Prioritized Experience Replay (PER DQN) <documents/api/agents/drl/perdqn_agent>`.
* :class:`NoisyDQN_Agent` : :doc:`DQN with Noisy Layers (Noisy DQN) <documents/api/agents/drl/noisydqn_agent>`.
* :class:`DRQN_Agent` : :doc:`Deep Recurrent Q-Network (DRQN) <documents/api/agents/drl/drqn_agent>`.
* :class:`QRDQN_Agent` : :doc:`DQN with Quantile Regression (QR-DQN) <documents/api/agents/drl/qrdqn_agent>`.
* :class:`C51_Agent` : :doc:`Categorical 51 DQN (C51) <documents/api/agents/drl/c51_agent>`.

**Policy-based:**

* :class:`PG_Agent` : :doc:`Policy Gradient (PG) <documents/api/agents/drl/pg_agent>`.
* :class:`NPG_Agent` : :doc:`Natural Policy Gradient (NPG) <documents/api/agents/drl/npg_agent>`.
* :class:`PPG_Agent` : :doc:`Phasic Policy Gradient (PPG) <documents/api/agents/drl/ppg_agent>`.
* :class:`A2C_Agent` : :doc:`Advantage Actor Critic (A2C) <documents/api/agents/drl/a2c_agent>`.
* :class:`SAC_Agent` : :doc:`Soft Actor-Critic (SAC) <documents/api/agents/drl/sac_agent>`.
* :class:`PPOCLIP_Agent` : :doc:`Proximal Policy Optimization with Clipped Objective (PPO-Clip) <documents/api/agents/drl/ppoclip_agent>`.
* :class:`PPOKL_Agent` : :doc:`Proximal Policy Optimization with KL Divergence (PPO-KL) <documents/api/agents/drl/ppokl_agent>`.
* :class:`DDPG_Agent` : :doc:`Deep Deterministic Policy Gradient (DDPG) <documents/api/agents/drl/ddpg_agent>`.
* :class:`TD3_Agent` : :doc:`Twin Delayed Deep Deterministic Policy Gradient (TD3) <documents/api/agents/drl/td3_agent>`.
* :class:`PDQN_Agent` : :doc:`Parameterised Deep Q-Network (P-DQN) <documents/api/agents/drl/pdqn_agent>`.
* :class:`MPDQN_Agent` : :doc:`Multi-pass Parameterised Deep Q-Network (MP-DQN) <documents/api/agents/drl/mpdqn_agent>`.
* :class:`SPDQN_Agent` : :doc:`Split parameterised Deep Q-Network (SP-DQN) <documents/api/agents/drl/spdqn_agent>`.

**MARL-based:**

* :class:`IQL_Agents` : :doc:`Independent Q-Learning (IQL) <documents/api/agents/marl/iql_agents>`.
* :class:`VDN_Agents` : :doc:`Value Decomposition Networks (VDN) <documents/api/agents/marl/vdn_agents>`.
* :class:`QMIX_Agents` : :doc:`Q-Mixing Networks (QMIX) <documents/api/agents/marl/qmix_agents>`.
* :class:`WQMIX_Agents` : :doc:`Weighted Q-Mixing Networks (WQMIX) <documents/api/agents/marl/wqmix_agents>`.
* :class:`QTRAN_Agents` : :doc:`Q-Transformation (QTRAN) <documents/api/agents/marl/qtran_agents>`.
* :class:`DCG_Agents` : :doc:`Deep Coordination Graphs (DCG) <documents/api/agents/marl/dcg_agents>`.
* :class:`IDDPG_Agents` : :doc:`Independent Deep Deterministic Policy Gradient (IDDPG) <documents/api/agents/marl/iddpg_agents>`.
* :class:`MADDPG_Agents` : :doc:`Multi-agent Deep Deterministic Policy Gradient (MADDPG) <documents/api/agents/marl/maddpg_agents>`.
* :class:`IAC_Agents` : :doc:`Independent Actor-Critic (IAC) <documents/api/agents/marl/iac_agents>`.
* :class:`COMA_Agents` : :doc:`Counterfactual Multi-agent Policy Gradient (COMA) <documents/api/agents/marl/coma_agents>`.
* :class:`VDAC_Agents` : :doc:`Value-Decomposition Actor-Critic (VDAC) <documents/api/agents/marl/vdac_agents>`.
* :class:`IPPO_Agents` : :doc:`Independent Proximal Policy Optimization (IPPO) <documents/api/agents/marl/ippo_agents>`.
* :class:`MAPPO_Agents` : :doc:`Multi-agent Proximal Policy Optimization (MAPPO) <documents/api/agents/marl/mappo_agents>`.
* :class:`MFQ_Agents` : :doc:`Mean-Field Q-Learning (MFQ) <documents/api/agents/marl/mfq_agents>`.
* :class:`MFAC_Agents` : :doc:`Mean-Field Actor-Critic (MFAC) <documents/api/agents/marl/mfac_agents>`.
* :class:`ISAC_Agents` : :doc:`Independent Soft Actor-Critic (ISAC) <documents/api/agents/marl/isac_agents>`.
* :class:`MASAC_Agents` : :doc:`Multi-agent Soft Actor-Critic (MASAC) <documents/api/agents/marl/masac_agents>`.
* :class:`MATD3_Agents` : :doc:`Multi-agent Twin Delayed Deep Deterministic Policy Gradient (MATD3) <documents/api/agents/marl/matd3_agents>`.
* :class:`IC3Net_Agents` : :doc:`Individual Controlled Continuous Communication Model (IC3Net) <documents/api/agents/marl/ic3net_agents>`.

“玄策”整体框架
------------------------------------------

“玄策”的整体框架如下图所示.

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

   documents/api/agents
   documents/api/environments
   documents/api/configs
   documents/api/runners
   documents/api/representations
   documents/api/policies
   documents/api/learners
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
   documents/CONTRIBUTING
   Contribute to the Docs (EN) <https://github.com/agi-brain/xuance/tree/master/docs>
   Contribute to the Docs (CN) <https://github.com/agi-brain/xuance-docs-zh_CN/tree/master/docs>
