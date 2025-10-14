自定义算法：MARL
=========================================================

.. raw:: html

   <a href="https://colab.research.google.com/github/agi-brain/xuance/blob/master/docs/source/notebook-colab/new_algorithm.ipynb"
      target="_blank"
      rel="noopener noreferrer"
      style="float: left; margin-left: 0px;">
       <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
   </a>
   <br>

用户可以在玄策框架所包含的算法之外创建自己的多智能体强化学习算法。

本教程将引导您完成使用玄策框架创建、训练和测试自定义 MARL 算法的全过程。
示例演示了如何定义自定义策略、学习器和智能体，同时利用玄策的模块化架构进行强化学习实验。

步骤 1: 定义策略模块
-------------------------------------------------------------

策略 policy 是智能体的决策模块，它将从环境中获取的观测信息映射至动作空间。这里，我们给出了一个自定义策略（MyPolicy）的示例：

.. code-block:: python

    class MyMARLPolicy(nn.Module):
        """
        An example of self-defined multi-agent policy for Independent DQN learning.

        Args:
            action_space: The action space of the environment.
            n_agents: The number of agents.
            representation: A neural network module responsible for extracting meaningful features.
            hidden_dim: Specifies the number of units in each hidden layer.
            device: The calculating device.
            use_parameter_sharing: Whether to share parameters across agents.
            model_keys: The keys for the models (agent names).
        """

        def __init__(self, action_space, n_agents, representation, hidden_dim, device,
                     use_parameter_sharing=True, model_keys=None, **kwargs):
            super(MyMARLPolicy, self).__init__()
            self.action_space = action_space
            self.n_agents = n_agents
            self.use_parameter_sharing = use_parameter_sharing
            self.device = device
            self.model_keys = model_keys or [f"agent_{i}" for i in range(n_agents)]

            # Build representations and Q-networks for each agent
            if use_parameter_sharing:
                # All agents share the same parameters
                self.representation = representation
                # Get feature dimension from the first model key
                self.feature_dim = self.representation[self.model_keys[0]].output_shapes['state'][0]

                # Single shared Q-network
                self.q_net = nn.Sequential(
                    nn.Linear(self.feature_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_space[self.model_keys[0]].n),
                ).to(device)
                self.target_q_net = deepcopy(self.q_net)
            else:
                # Each agent has its own parameters
                self.representations = representation  # representation is already a ModuleDict
                self.q_nets = nn.ModuleDict()
                self.target_q_nets = nn.ModuleDict()

                for key in self.model_keys:
                    feature_dim = self.representations[key].output_shapes['state'][0]

                    self.q_nets[key] = nn.Sequential(
                        nn.Linear(feature_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, action_space[key].n),
                    ).to(device)
                    self.target_q_nets[key] = deepcopy(self.q_nets[key])

        def forward(self, observation, agent_ids=None, avail_actions=None, **kwargs):
            """
            Forward pass of the policy network.

            Args:
                observation: Dict of observations for each agent.
                agent_ids: Agent identifiers (optional).
                avail_actions: Available actions mask (optional).
                **kwargs: Additional keyword arguments.

            Returns:
                Tuple containing:
                    - outputs: Feature representations for each agent
                    - argmax_actions: Greedy actions for each agent
                    - q_values: Q-values for each agent
            """
            batch_size = list(observation.values())[0].shape[0]
            outputs = {}
            argmax_actions = {}
            q_values = {}

            if self.use_parameter_sharing:
                # Use shared parameters for all agents
                model_key = self.model_keys[0]  # Get the shared model key
                for agent_key in observation.keys():
                    obs_out = self.representation[model_key](observation[agent_key])
                    q_out = self.q_net(obs_out['state'])
                    argmax_action = q_out.argmax(dim=-1)

                    outputs[agent_key] = obs_out
                    argmax_actions[agent_key] = argmax_action
                    q_values[agent_key] = q_out
            else:
                # Use separate parameters for each agent
                for key in self.model_keys:
                    obs_out = self.representations[key](observation[key])
                    q_out = self.q_nets[key](obs_out['state'])
                    argmax_action = q_out.argmax(dim=-1)

                    outputs[key] = obs_out
                    argmax_actions[key] = argmax_action
                    q_values[key] = q_out

            return outputs, argmax_actions, q_values

        def target(self, observation, agent_ids=None, **kwargs):
            """
            Forward pass using target networks.

            Args:
                observation: Dict of observations for each agent.
                agent_ids: Agent identifiers (optional).
                **kwargs: Additional keyword arguments.

            Returns:
                Tuple containing:
                    - outputs: Feature representations for each agent
                    - argmax_actions: Target greedy actions for each agent
                    - q_targets: Target Q-values for each agent
            """
            batch_size = list(observation.values())[0].shape[0]
            outputs = {}
            argmax_actions = {}
            q_targets = {}

            if self.use_parameter_sharing:
                model_key = self.model_keys[0]  # Get the shared model key
                for agent_key in observation.keys():
                    obs_out = self.representation[model_key](observation[agent_key])
                    q_target = self.target_q_net(obs_out['state'])
                    argmax_action = q_target.argmax(dim=-1)

                    outputs[agent_key] = obs_out
                    argmax_actions[agent_key] = argmax_action.detach()
                    q_targets[agent_key] = q_target.detach()
            else:
                for key in self.model_keys:
                    obs_out = self.representations[key](observation[key])
                    q_target = self.target_q_nets[key](obs_out['state'])
                    argmax_action = q_target.argmax(dim=-1)

                    outputs[key] = obs_out
                    argmax_actions[key] = argmax_action.detach()
                    q_targets[key] = q_target.detach()

            return outputs, argmax_actions, q_targets

        def copy_target(self):
            """Reset the parameters of target Q network as the Q network."""
            if self.use_parameter_sharing:
                for ep, tp in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                    tp.data.copy_(ep)
            else:
                for key in self.model_keys:
                    for ep, tp in zip(self.q_nets[key].parameters(), self.target_q_nets[key].parameters()):
                        tp.data.copy_(ep)


关键点：

- 表征器（representation）：用于提取状态特征，将环境表征与Q值计算解耦。
- 网络（networks）：策略使用前馈神经网络来计算动作并估计Q值。
- 设备（device）：需指定计算设备，CPU或GPU，GPU编号等。

步骤 2: 定义学习器模块（Learner）
-------------------------------------------------------------

学习器（Learner）主要负责定义优化器、确定优化目标，从而计算出损失函数，完成反向传播，从而更新策略模块的网络参数。

.. code-block:: python

    class MyMARLLearner(LearnerMAS):
        """
        Custom multi-agent learner implementing Independent DQN learning.

        This learner extends the base LearnerMAS class to provide custom
        implementation for multi-agent Q-learning with independent agents.
        """

        def __init__(self, config, model_keys, agent_keys, policy, callback):
            super(MyMARLLearner, self).__init__(config, model_keys, agent_keys, policy, callback)
            # Build the optimizer.
            self.optimizer = torch.optim.Adam(self.policy.parameters(), self.config.learning_rate, eps=1e-5)
            self.loss = nn.MSELoss()  # Build a loss function
            self.sync_frequency = config.sync_frequency  # The period to synchronize the target network

        def update(self, sample):
            """
            Update the policy networks using a batch of training samples.

            Args:
                sample: Dictionary containing training batch data with keys:
                    - obs: Current observations for all agents
                    - actions: Actions taken by all agents
                    - obs_next: Next observations for all agents
                    - rewards: Rewards received by all agents
                    - terminals: Terminal flags for all agents

            Returns:
                Dict containing training information and losses.
            """
            info = {}
            self.iterations += 1

            # Get a batch of training samples for all agents
            # Use the actual keys from the sample data
            actual_agent_keys = list(sample['obs'].keys())
            obs_batch = {key: torch.as_tensor(sample['obs'][key], device=self.device) for key in actual_agent_keys}
            act_batch = {key: torch.as_tensor(sample['actions'][key], device=self.device) for key in actual_agent_keys}
            next_batch = {key: torch.as_tensor(sample['obs_next'][key], device=self.device) for key in actual_agent_keys}
            rew_batch = {key: torch.as_tensor(sample['rewards'][key], device=self.device) for key in actual_agent_keys}
            ter_batch = {key: torch.as_tensor(sample['terminals'][key], dtype=torch.float, device=self.device) for key in actual_agent_keys}

            # Forward passes for all agents
            _, _, q_eval = self.policy(obs_batch)
            _, _, q_next = self.policy.target(next_batch)

            # Compute losses for all agents
            total_loss = 0
            agent_losses = {}

            for key in actual_agent_keys:
                # Now each agent has its own Q values in the output
                q_next_action = q_next[key].max(dim=-1).values
                q_eval_action = q_eval[key].gather(-1, act_batch[key].long().unsqueeze(-1)).reshape(-1)
                target_value = rew_batch[key] + (1 - ter_batch[key]) * self.gamma * q_next_action

                # Compute loss for this agent
                agent_loss = self.loss(q_eval_action, target_value.detach())
                agent_losses[key] = agent_loss.item()
                total_loss += agent_loss

            # Backward and optimizing steps
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Synchronize the target network
            if self.iterations % self.sync_frequency == 0:
                self.policy.copy_target()

            # Set the variables you need to observe
            info.update({
                'total_loss': total_loss.item(),
                'iterations': self.iterations,
            })

            # Add individual agent losses
            for key in actual_agent_keys:
                info[f'loss_{key}'] = agent_losses[key]
                info[f'q_eval_{key}'] = q_eval[key].mean().item()

            return info

关键要点:

- 优化器（optimizer）: 优化器的选择需在学习器的 ``__init__`` 方法中定义.
- 更新方法（update）: 在该方法中，传入一个batch的经验数据，完成前向传播并计算出损失函数，最后完成反向传播和参数更新.
- info字典: 在 ``info`` 字典中写入您想在训练过程中观察的变量.

步骤 3: 定义智能体模块（Agent)
-------------------------------------------------------------

智能体模块包含了 ``policy``，``learner``，``environment``等模块，用于实现智能体和环境的交互过程。

.. code-block:: python

    class MyMARLAgents(OffPolicyMARLAgents):
        """Multi-agent version of the custom DQN implementation."""

        def __init__(self, config: Namespace,
                     envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv],
                     callback: Optional[BaseCallback] = None):
            super(MyMARLAgents, self).__init__(config, envs, callback)

            # Initialize epsilon-greedy parameters
            self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
            self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy
            self.e_greedy = self.start_greedy

            self.policy = self._build_policy()  # Build the policy module
            self.memory = self._build_memory()  # Build the replay buffer
            REGISTRY_Learners['MyMARLLearner'] = MyMARLLearner  # Registry your pre-defined learner
            self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy, self.callback)  # Build the learner

        def _build_policy(self) -> Module:
            """
            Build multi-agent policy.

            Constructs the custom multi-agent policy with appropriate representation
            networks and Q-networks based on configuration settings.

            Returns:
                Module: The constructed multi-agent policy.
            """
            normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
            initializer = torch.nn.init.orthogonal_
            activation = ActivationFunctions[self.config.activation]
            device = self.device

            # Build representation
            representation = self._build_representation(self.config.representation, self.observation_space, self.config)

            # Build custom multi-agent policy
            policy = MyMARLPolicy(
                action_space=self.action_space,
                n_agents=self.n_agents,
                representation=representation,
                hidden_dim=64,  # You can make this configurable
                device=device,
                use_parameter_sharing=self.use_parameter_sharing,
                model_keys=self.model_keys
            )
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
        config = get_configs(file_dir="new_marl.yaml")  # Get the config settings from .yaml file
        config = Namespace(**config)  # Convert the config from dict to argparse
        envs = make_envs(config)  # Make vectorized multi-agent environments
        agents = MyMARLAgents(config, envs)  # Instantiate your pre-build multi-agent class

        if not config.test_mode:  # Training mode.
            train_steps = config.running_steps // config.parallels
            agents.train(train_steps)  # Train your agents
            agents.save_model("final_train_model.pth")  # After training, save the model
        else:  # Testing mode.
            def env_fn():
                config.parallels = config.test_episode
                return make_envs(config)

            agents.load_model(path=agents.model_dir_load)  # Load pre-trained model
            scores = agents.test(env_fn, config.test_episode)  # Test your agents
            print(f"Mean Score: {np.array(scores).mean()}, Std: {np.array(scores).std()}")

        agents.finish()  # Finish the agents
        envs.close()  # Close the environments

该示例的源码文件请参考以下链接:

`https://github.com/agi-brain/xuance/blob/master/examples/new_algorithm/new_marl.py <https://github.com/agi-brain/xuance/blob/master/examples/new_algorithm/new_marl.py>`_
