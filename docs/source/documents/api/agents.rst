Agents
======================

.. toctree::
  :hidden:

  Agent <agents/drl/basic_drl_class>
  MARLAgents <agents/marl/basic_marl_class>
  DQN_Agent <agents/drl/dqn>
  C51_Agent <agents/drl/c51>
  DDQN_Agent <agents/drl/ddqn>
  DuelDQN_Agent <agents/drl/dueldqn>
  NoisyDQN_Agent <agents/drl/noisydqn>
  PerDQN_Agent <agents/drl/perdqn>
  QRDQN_Agent <agents/drl/qrdqn>
  PG_Agent <agents/drl/pg>
  PPG <agents/drl/ppg>
  PPOCLIP_Agent <agents/drl/ppo_clip>
  PPOCKL_Agent <agents/drl/ppo_kl>
  PDQN_Agent <agents/drl/pdqn>
  SPDQN_Agent <agents/drl/spdqn>
  MPDQN_Agent <agents/drl/mpdqn>
  A2C_Agent <agents/drl/a2c>
  SAC_Agent <agents/drl/sac>
  SACDIS_Agent <agents/drl/sac_dis>
  DDPG_Agent <agents/drl/ddpg>
  TD3_Agent <agents/drl/td3>

  IQL_Agent <agents/marl/iql>
  VDN_Agent <agents/marl/vdn>
  QMIX_Agent <agents/marl/qmix>
  WQMIX_Agent <agents/marl/wqmix>
  QTRAN_Agent <agents/marl/qtran>
  DCG_Agent <agents/marl/dcg>
  IDDPG_Agent <agents/marl/iddpg>
  MADDPG_Agent <agents/marl/maddpg>
  ISAC_Agent <agents/marl/isac>
  MASAC_Agent <agents/marl/masac>
  IPPO_Agent <agents/marl/ippo>
  MAPPO_Agent <agents/marl/mappo>
  MATD3_Agent <agents/marl/matd3>
  VDAC <agents/marl/vdac>
  COMA_Agent <agents/marl/coma>
  MFQ_Agent <agents/marl/mfq>
  MFAC_Agent <agents/marl/mfac>

.. raw:: html

   <br><hr>

强化学习Agents（智能体）是能够与环境进行交互的、具有自主决策能力和自主学习能力的独立单元。
在与环境交互过程中，Agents获取观测信息，根据观测信息计算出动作信息并执行该动作，使得环境进入下一步状态。
通过不断和环境进行交互，Agents收集经验数据，再根据经验数据训练模型，从而获得更优的策略。
以下列出了“玄策”平台中包含的单智能体强化学习Agents。


.. list-table:: 
   :header-rows: 1

   * - Agents
     - PyTorch 
     - TensorFlow
     - MindSpore
   * - :doc:`DQN <agents/drl/dqn>`: Deep Q-Networks
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`C51DQN <agents/drl/c51>`: Distributional Reinforcement Learning
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Double DQN <agents/drl/ddqn>`: DQN with Double Q-learning
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Dueling DQN <agents/drl/dueldqn>`: DQN with Dueling network
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Noisy DQN <agents/drl/noisydqn>`: DQN with Parameter Space Noise
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`PERDQN <agents/drl/perdqn>`: DQN with Prioritized Experience Replay
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`QRDQN <agents/drl/qrdqn>`: DQN with Quantile Regression
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`VPG <agents/drl/pg>`: Vanilla Policy Gradient
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`PPG <agents/drl/ppg>`: Phasic Policy Gradient
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`PPO <agents/drl/ppo_clip>`: Proximal Policy Optimization
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`PDQN <agents/drl/pdqn>`: Parameterised DQN
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`SPDQN <agents/drl/spdqn>`: Split PDQN
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`MPDQN <agents/drl/mpdqn>`: Multi-pass PDQN
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`A2C <agents/drl/a2c>`: Advantage Actor Critic
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`SAC <agents/drl/sac>`: Soft Actor-Critic
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`SAC-Dis <agents/drl/sac_dis>`: SAC for Discrete Actions
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`DDPG <agents/drl/ddpg>`: Deep Deterministic Policy Gradient
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`TD3 <agents/drl/td3>`: Twin Delayed DDPG
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`

.. raw:: html

   <br><hr>
