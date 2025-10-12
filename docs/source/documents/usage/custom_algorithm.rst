自定义算法
=========================================================

.. raw:: html

   <a href="https://colab.research.google.com/github/agi-brain/xuance/blob/master/docs/source/notebook-colab/new_algorithm.ipynb"
      target="_blank"
      rel="noopener noreferrer"
      style="float: left; margin-left: 0px;">
       <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
   </a>
   <br>

用户可以在玄策框架的默认算法之外创建自己的新算法。

本教程将引导您完成使用玄策框架创建、训练和测试自定义离策略强化学习（RL）智能体的全过程。
示例演示了如何定义自定义策略、学习器和智能体，同时利用玄策的模块化架构进行强化学习实验。

步骤 1: 定义策略模块
-------------------------------------------------------------

策略 policy 是智能体的决策模块，它将从环境中获取的观测信息映射至动作空间。这里，我们给出了一个自定义策略（MyPolicy）的示例：

.. code-block:: python

    class MyPolicy(nn.Module):
    """
    An example of self-defined policy.

    Args:
        representation (nn.Module): A neural network module responsible for extracting meaningful features from the raw observations provided by the environment.
        hidden_dim (int): Specifies the number of units in each hidden layer, determining the model’s capacity to capture complex patterns.
        n_actions (int): The total number of discrete actions available to the agent in the environment.
        device (torch.device): The calculating device.


    Note: The inputs to the __init__ method are not rigidly defined. You can extend or modify them as needed to accommodate additional settings or configurations specific to your application.
    """

        def __init__(self, representation: nn.Module, hidden_dim: int, n_actions: int, device: torch.device):
            super(MyPolicy, self).__init__()
            self.representation = representation  # Specify the representation.
            self.feature_dim = self.representation.output_shapes['state'][0]  # Dimension of the representation's output.
            self.q_net = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions),
            ).to(device)  # The Q network.
            self.target_q_net = deepcopy(self.q_net)  # Target Q network.

        def forward(self, observation):
            output_rep = self.representation(observation)  # Get the output of the representation module.
            output = self.q_net(output_rep['state'])  # Get the output of the Q network.
            argmax_action = output.argmax(dim=-1)  # Get greedy actions.
            return output_rep, argmax_action, output

        def target(self, observation):
            outputs_target = self.representation(observation)  # Get the output of the representation module.
            Q_target = self.target_q_net(outputs_target['state'])  # Get the output of the target Q network.
            argmax_action = Q_target.argmax(dim=-1)  # Get greedy actions that output by target Q network.
            return outputs_target, argmax_action.detach(), Q_target.detach()

        def copy_target(self):  # Reset the parameters of target Q network as the Q network.
            for ep, tp in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                tp.data.copy_(ep)


关键点：

- 表征器（representation）：用于提取状态特征，将环境表征与Q值计算解耦。
- 网络（networks）：策略使用前馈神经网络来计算动作并估计Q值。
- 设备（device）：需指定计算设备，CPU或GPU，GPU编号等。

步骤 2: 定义学习器模块（Learner）
-------------------------------------------------------------

学习器（Learner）主要负责定义优化器、确定优化目标，从而计算出损失函数，完成反向传播，从而更新策略模块的网络参数。

.. code-block:: python

    class MyLearner(Learner):
        def __init__(self, config, policy):
            super(MyLearner, self).__init__(config, policy)
            # Build the optimizer.
            self.optimizer = torch.optim.Adam(self.policy.parameters(), self.config.learning_rate, eps=1e-5)
            self.loss = nn.MSELoss()  # Build a loss function.
            self.sync_frequency = config.sync_frequency  # The period to synchronize the target network.

        def update(self, **samples):
            info = {}
            self.iterations += 1
            '''Get a batch of training samples.'''
            obs_batch = torch.as_tensor(samples['obs'], device=self.device)
            act_batch = torch.as_tensor(samples['actions'], device=self.device)
            next_batch = torch.as_tensor(samples['obs_next'], device=self.device)
            rew_batch = torch.as_tensor(samples['rewards'], device=self.device)
            ter_batch = torch.as_tensor(samples['terminals'], dtype=torch.float, device=self.device)

            # Feedforward steps.
            _, _, q_eval = self.policy(obs_batch)
            _, _, q_next = self.policy.target(next_batch)
            q_next_action = q_next.max(dim=-1).values
            q_eval_action = q_eval.gather(-1, act_batch.long().unsqueeze(-1)).reshape(-1)
            target_value = rew_batch + (1 - ter_batch) * self.gamma * q_next_action
            loss = self.loss(q_eval_action, target_value.detach())

            # Backward and optimizing steps.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Synchronize the target network
            if self.iterations % self.sync_frequency == 0:
                self.policy.copy_target()

            # Set the variables you need to observe.
            info.update({'loss': loss.item(),
                         'iterations': self.iterations,
                         'q_eval_action': q_eval_action.mean().item()})

            return info

关键要点:

- 优化器（optimizer）: 优化器的选择需在学习器的 ``__init__`` 方法中定义.
- 更新方法（update）: 在该方法中，传入一个batch的经验数据，完成前向传播并计算出损失函数，最后完成反向传播和参数更新.
- info字典: 在 ``info`` 字典中写入您想在训练过程中观察的变量.

步骤 3: 定义智能体模块（Agent)
-------------------------------------------------------------

智能体模块包含了 ``policy``，``learner``，``environment``等模块，用于实现智能体和环境的交互过程。

.. code-block:: python

    class MyAgent(OffPolicyAgent):
        def __init__(self, config, envs):
            super(MyAgent, self).__init__(config, envs)
            self.policy = self._build_policy()  # Build the policy module.
            self.memory = self._build_memory()  # Build the replay buffer.
            REGISTRY_Learners['MyLearner'] = MyLearner  # Registry your pre-defined learner.
            self.learner = self._build_learner(self.config, self.policy)  # Build the learner.

        def _build_policy(self):
            # First create the representation module.
            representation = self._build_representation("Basic_MLP", self.observation_space, self.config)
            # Build your customized policy module.
            policy = MyPolicy(representation, 64, self.action_space.n, self.config.device)
            return policy

关键要点:

- 策略（policy）: 在 ``_build_policy`` 方法中创建表征器模块，然后创建策略模块.
- 经验回放池（memory）: 在 ``_build_memory`` 方法中创建经验回放池，用于存储经验数据.
- 学习器（learner）: 在 ``_build_learner`` 方法中创建学习器模块.

步骤 4: 创建智能体模块并运行
-------------------------------------------------------------

在准备好以上各模块之后，在主程序中获取 ``config`` 参数配置，创建环境、智能体模块，
利用 ``Agent`` 模块中预定义的 ``train``，``test`` 方法，完成训练和测试。

.. code-block:: python

    if __name__ == '__main__':
        config = get_configs(file_dir="./new_rl.yaml")  # Get the config settings from .yaml file.
        config = Namespace(**config)  # Convert the config from dict to argparse.
        envs = make_envs(config)  # Make vectorized environments.
        agent = MyAgent(config, envs)  # Instantiate your pre-build agent class.

        if not config.test_mode:  # Training mode.
            agent.train(config.running_steps // envs.num_envs)  # Train your agent.
            agent.save_model("final_train_model.pth")  # After training, save the model.
        else:  # Testing mode.
            config.parallels = 1  # Test on one environment.
            env_fn = lambda: make_envs(config)  # The method to create testing environment.
            agent.load_model(agent.model_dir_load)  # Load pre-trained model.
            scores = agent.test(env_fn, config.test_episode)  # Test your agent.

        agent.finish()  # Finish the agent.
        envs.close()  # Close the environments.

该示例的源码文件请参考以下链接:

`https://github.com/agi-brain/xuance/blob/master/examples/new_algorithm/new_rl.py <https://github.com/agi-brain/xuance/blob/master/examples/new_algorithm/new_rl.py>`_
