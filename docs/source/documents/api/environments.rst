Environments
======================

本软件包含的单智能体任务仿真环境有gym下的Atari，Mujoco，Classic Control，Box2D。
包含的多智能体任务仿真环境有PettingZoo开源环境下的MPE和SISL，StarCraft2，MAgent2，Google Football环境等。
每个仿真环境下包含丰富的任务场景，如下表所示。


.. toctree::
    :hidden:

    Single-Agent Env <environments/single_agent_env>
    Multi-Agent Env <environments/multi_agent_env>
    vectorization <environments/vector_envs>
    utils <environments/utils>

* :doc:`Single-Agent Env <environments/single_agent_env>`.
* :doc:`Multi-Agent Env <environments/multi_agent_env>`.
* :doc:`vectorization <environments/vector_envs>`.
* :doc:`utils <environments/utils>`.

Make Environment
-------------------------

.. automodule:: xuance.environment
    :members: make_envs
    :undoc-members:
    :show-inheritance:
