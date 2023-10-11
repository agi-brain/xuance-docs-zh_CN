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
   * - :doc:`Deep Q-Networks <agents/drl/dqn>` (DQN)
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Distributional Reinforcement Learning <agents/drl/c51>` (C51DQN)
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`DQN with Double Q-learning <agents/drl/ddqn>` (Double DQN)
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`DQN with Dueling network <agents/drl/dueldqn>` (Dueling DQN)
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`DQN with Parameter Space Noise <agents/drl/noisydqn>` (Noisy DQN)
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`DQN with Prioritized Experience Replay <agents/drl/perdqn>` (PERDQN)
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`DQN with Quantile Regression <agents/drl/qrdqn>` (QRDQN)
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Vanilla Policy Gradient <agents/drl/pg>` (VPG)
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Phasic Policy Gradient <agents/drl/ppg>` (PPG)
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Proximal Policy Optimization <agents/drl/ppo_clip>` (PPO)
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Parameterised DQN <agents/drl/pdqn>` (PDQN)
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Split PDQN <agents/drl/spdqn>` (SPDQN)
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Multi-pass PDQN <agents/drl/mpdqn>` (MPDQN)
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Advantage Actor Critic <agents/drl/a2c>` (A2C)
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Soft Actor-Critic <agents/drl/sac>` (SAC)
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`SAC for Discrete Actions <agents/drl/sac_dis>` (SAC-Dis)
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Deep Deterministic Policy Gradient <agents/drl/ddpg>` (DDPG)
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Twin Delayed DDPG <agents/drl/td3>` (TD3)
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`

.. raw:: html

   <br><hr>
