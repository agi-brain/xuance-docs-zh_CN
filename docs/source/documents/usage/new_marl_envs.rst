多智能体环境构建
---------------------------------

在 XuanCe 中，用户同样可以灵活地创建并运行自己定制的多智能体（multi-agent）环境，
除了使用内置的环境之外，还可以在此基础上扩展出包含多个智能体的复杂交互场景。

.. raw:: html

   <a href="https://colab.research.google.com/github/agi-brain/xuance/blob/master/docs/source/notebook-colab/new_marl_envs.ipynb"
      target="_blank"
      rel="noopener noreferrer"
      style="float: left; margin-left: 0px;">
       <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
   </a>
   <br>

步骤 1：创建新的多智能体环境类
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

首先，你需要准备一个原始环境，即“部分可观测马尔可夫决策过程”（POMDP）。
然后，基于 XuanCe 的基础类 ``RawMultiAgentEnv`` 定义一个新的环境。

以下是一个示例：

.. code-block:: python

    import numpy as np
    from gymnasium.spaces import Box
    from xuance.environment import RawMultiAgentEnv

    class MyNewMultiAgentEnv(RawMultiAgentEnv):
        def __init__(self, env_config):
            super(MyNewMultiAgentEnv, self).__init__()
            self.env_id = env_config.env_id
            self.num_agents = 3
            self.agents = [f"agent_{i}" for i in range(self.num_agents)]
            self.state_space = Box(-np.inf, np.inf, shape=[8, ])
            self.observation_space = {agent: Box(-np.inf, np.inf, shape=[8, ]) for agent in self.agents}
            self.action_space = {agent: Box(-np.inf, np.inf, shape=[2, ]) for agent in self.agents}
            self.max_episode_steps = 25
            self._current_step = 0

        def get_env_info(self):
            return {'state_space': self.state_space,
                    'observation_space': self.observation_space,
                    'action_space': self.action_space,
                    'agents': self.agents,
                    'num_agents': self.num_agents,
                    'max_episode_steps': self.max_episode_steps}

        def avail_actions(self):
            return None

        def agent_mask(self):
            """Returns boolean mask variables indicating which agents are currently alive."""
            return {agent: True for agent in self.agents}

        def state(self):
            """Returns the global state of the environment."""
            return self.state_space.sample()

        def reset(self):
            observation = {agent: self.observation_space[agent].sample() for agent in self.agents}
            info = {}
            self._current_step = 0
            return observation, info

        def step(self, action_dict):
            self._current_step += 1
            observation = {agent: self.observation_space[agent].sample() for agent in self.agents}
            rewards = {agent: np.random.random() for agent in self.agents}
            terminated = {agent: False for agent in self.agents}
            truncated = False if self._current_step < self.max_episode_steps else True
            info = {}
            return observation, rewards, terminated, truncated, info

        def render(self, *args, **kwargs):
            return np.ones([64, 64, 64])

        def close(self):
            return


步骤 2：创建配置文件并读取配置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

接下来，你需要按照 :doc:`“进一步使用” <further_usage>` 中的步骤 1 创建一个 YAML 配置文件。

以下是一个为 DDPG 算法准备的配置示例，文件名为：``ippo_new_configs.yaml``。

.. code-block:: python

    dl_toolbox: "torch"  # The deep learning toolbox. Choices: "torch", "mindspore", "tensorlayer"
    project_name: "XuanCe_Benchmark"
    logger: "tensorboard"  # Choices: tensorboard, wandb.
    wandb_user_name: "your_user_name"
    render: True
    render_mode: 'rgb_array' # Choices: 'human', 'rgb_array'.
    fps: 15
    test_mode: False
    device: "cpu"  # Choose an calculating device. PyTorch: "cpu", "cuda:0"; TensorFlow: "cpu"/"CPU", "gpu"/"GPU"; MindSpore: "CPU", "GPU", "Ascend", "Davinci".
    distributed_training: False  # Whether to use multi-GPU for distributed training.
    master_port: '12355'  # The master port for current experiment when use distributed training.

    agent: "IPPO"
    env_name: "MyNewMultiAgentEnv"
    env_id: "new_env_id"
    env_seed: 1
    continuous_action: True  # Continuous action space or not.
    learner: "IPPO_Learner"  # The learner name.
    policy: "Gaussian_MAAC_Policy"
    representation: "Basic_MLP"
    vectorize: "DummyVecMultiAgentEnv"

    # recurrent settings for Basic_RNN representation.
    use_rnn: False  # If to use recurrent neural network as representation. (The representation should be "Basic_RNN").
    rnn: "GRU"  # The type of recurrent layer.
    fc_hidden_sizes: [64, 64, 64]  # The hidden size of feed forward layer in RNN representation.
    recurrent_hidden_size: 64  # The hidden size of the recurrent layer.
    N_recurrent_layers: 1  # The number of recurrent layer.
    dropout: 0  # dropout should be a number in range [0, 1], the probability of an element being zeroed.
    normalize: "LayerNorm"  # Layer normalization.
    initialize: "orthogonal"  # Network initializer.
    gain: 0.01  # Gain value for network initialization.

    # recurrent settings for Basic_RNN representation.
    representation_hidden_size: [64, ]  # A list of hidden units for each layer of Basic_MLP representation networks.
    actor_hidden_size: [64, ]  # A list of hidden units for each layer of actor network.
    critic_hidden_size: [64, ]  # A list of hidden units for each layer of critic network.
    activation: "relu"  # The activation function of each hidden layer.
    activation_action: "sigmoid"  # The activation function for the last layer of the actor.
    use_parameter_sharing: True  # If to use parameter sharing for all agents' policies.
    use_actions_mask: False  # If to use actions mask for unavailable actions.

    seed: 1  # Random seed.
    parallels: 16  # The number of environments to run in parallel.
    buffer_size: 3200  # Number of the transitions (use_rnn is False), or the episodes (use_rnn is True) in replay buffer.
    n_epochs: 10  # Number of epochs to train.
    n_minibatch: 1 # Number of minibatch to sample and train.  batch_size = buffer_size // n_minibatch.
    learning_rate: 0.0007  # Learning rate.
    weight_decay: 0  # The steps to decay the greedy epsilon.

    vf_coef: 0.5  # Coefficient factor for critic loss.
    ent_coef: 0.01  # Coefficient factor for entropy loss.
    target_kl: 0.25  # For MAPPO_KL learner.
    clip_range: 0.2  # The clip range for ratio in MAPPO_Clip learner.
    gamma: 0.99  # Discount factor.

    # tricks
    use_linear_lr_decay: False  # If to use linear learning rate decay.
    end_factor_lr_decay: 0.5  # The end factor for learning rate scheduler.
    use_global_state: False  # If to use global state to replace merged observations.
    use_value_clip: True  # Limit the value range.
    value_clip_range: 0.2  # The value clip range.
    use_value_norm: True  # Use running mean and std to normalize rewards.
    use_huber_loss: True  # True: use huber loss; False: use MSE loss.
    huber_delta: 10.0  # The threshold at which to change between delta-scaled L1 and L2 loss. (For huber loss).
    use_advnorm: True  # If to use advantage normalization.
    use_gae: True  # Use GAE trick.
    gae_lambda: 0.95  # The GAE lambda.
    use_grad_clip: True  # Gradient normalization.
    grad_clip_norm: 10.0  # The max norm of the gradient.
    clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm().

    running_steps: 100000  # The total running steps.
    eval_interval: 10000  # The interval between every two trainings.
    test_episode: 5  # The episodes to test in each test period.

    log_dir: "./logs/ippo/"
    model_dir: "./models/ippo/"


然后，读取该配置文件:

.. code-block:: python

    import argparse
    from xuance.common import get_configs
    configs_dict = get_configs(file_dir="ippo_new_configs.yaml")
    configs = argparse.Namespace(**configs_dict)


步骤 3：将环境添加到注册表中
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在定义完一个新的环境类之后，需要将其添加到 ``REGISTRY_MULTI_AGENT_ENV`` 中进行注册。

.. code-block:: python

    from xuance.environment import REGISTRY_MULTI_AGENT_ENV
    REGISTRY_MULTI_AGENT_ENV[configs.env_name] = MyNewMultiAgentEnv


步骤 4：创建你的环境并在 XuanCe 中运行
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

现在，你可以创建自己的环境，并直接使用 XuanCe 提供的算法运行它。

以下是使用 IPPO 算法 的示例：

.. code-block:: python

    from xuance.environment import make_envs
    from xuance.torch.agents import IPPO_Agents

    envs = make_envs(configs)  # Make parallel environments.
    Agent = IPPO_Agents(config=configs, envs=envs)  # Create a DDPG agent from XuanCe.
    Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
    Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
    Agent.finish()  # Finish the training.


完整代码
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

上述步骤的完整代码可在以下链接查看： `https://github.com/agi-brain/xuance/blob/master/examples/new_environments/ippo_new_env.py <https://github.com/agi-brain/xuance/blob/master/examples/new_environments/ippo_new_env.py>`_
