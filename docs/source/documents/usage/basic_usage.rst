快速开始
=======================

运行一个DRL算法
-----------------------

安装完玄策后，用户只需要三行代码即可运行一个深度强化学习算法。首先，你需要通过指定 ``agent_name`` 和 ``env_name`` 等信息创建一个 *runner* 对象，
在 *runner* 的创建过程中，玄策已经创建好了 ``agent``, ``policy``, ``envs`` 等强化学习关键要素。接下来，只需要执行 ``runner.run()`` 就能实现算法的训练了。
一个DQN算法的实例如下：

:: 

    import xuanpolicy as xp
    runner = xp.get_runner(agent_name='dqn', 
                           env_name='toy/CartPole-v0', 
                           is_test=False)
    runner.run()

执行以上指令，终端将输出实验的基本信息和训练过程进度条，当进度条满格时表示训练结束，模型保存。

运行一个MARL算法
-----------------------

和单智能体强化学习不同的是，多智能体强化学习任务分为合作型、竞争型、以及合作竞争混合型。合作型任务的运行方法和前面介绍的相同，只需要指定算法名称和环境名称即可开始训练，例如：
:: 

    import xuanpolicy as xp
    runner = xp.get_runner(agent_name='maddpg', env_name='mpe/simple_spread', is_test=False)
    runner.run()

当环境中包含竞争型任务时，由于智能体的优化目标不同，玄策根据原始环境中的任务说明，将智能体进行分组，每组运行一个MARL算法。
例如，对于 `mpe/adversary <https://pettingzoo.farama.org/environments/mpe/simple_adversary/>`_ 环境，智能体集合为[adversary_0, agent_0,agent_1]。
玄策的环境封装模块将这些智能体分为两组，第一组为“adversary”智能体，第二组为“agent”智能体，每组均可指定一种MARL算法，实现方法如下：

:: 

    import xuanpolicy as xp
    runner = xp.get_runner(agent_name=["iddpg", "maddpg"], env_name='mpe/simple_push', is_test=False)
    runner.run()

在该示例中，第一组智能体使用IDDPG算法，而第二组使用MADDPG算法。执行以上命令后，终端将输出实验的基本信息和训练过程进度条，当进度条满格时表示训练结束，模型保存。

测试
-----------------------
完成算法训练后，玄策会在指定目录中保存模型文件和训练日志信息。用户可以通过指定runner.test=True来

:: 

    import xuanpolicy as xp
    runner_test = xp.get_runner(agent_name='dqn', env_name='toy/CartPole-v0', is_test=True)
    runner_test.run()



日志
-----------------------

用户可利用tensorboard工具来可视化训练过程。
当完成模型训练后，日志文件存放于根目录下的log文件夹中，具体路径根据用户的实际配置查找。
以./logs/dqn/torch/CartPole-v0路径为例，用户可通过以下指令实现日志可视化：
:: 
    
    tensorboard --logdir ./logs/dqn/torch/CartPole-v0

