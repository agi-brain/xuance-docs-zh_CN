自定义环境（单智能体）
---------------------------------

在 XuanCe 中，用户不仅可以使用内置的环境，还可以灵活地创建并运行自己定制的环境。

.. raw:: html

   <a href="https://colab.research.google.com/github/agi-brain/xuance/blob/master/docs/source/notebook-colab/new_drl_envs.ipynb"
      target="_blank"
      rel="noopener noreferrer"
      style="float: left; margin-left: 0px;">
       <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
   </a>
   <br>

步骤 1：创建一个新环境类
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

首先，你需要准备一个原始环境，也就是一个马尔可夫决策过程（Markov Decision Process, MDP）。
然后，基于 XuanCe 的基础类 ``RawEnvironment`` 来定义一个新的环境。

以下是一个示例：

.. code-block:: python

    import numpy as np
    from gymnasium.spaces import Box
    from xuance.environment import RawEnvironment

    class MyNewEnv(RawEnvironment):
        def __init__(self, env_config):
            super(MyNewEnv, self).__init__()
            self.env_id = env_config.env_id  # The environment id.
            self.observation_space = Box(-np.inf, np.inf, shape=[18, ])  # Define observation space.
            self.action_space = Box(-np.inf, np.inf, shape=[5, ])  # Define action space. In this example, the action space is continuous.
            self.max_episode_steps = 32  # The max episode length.
            self._current_step = 0  # The count of steps of current episode.

        def reset(self, **kwargs):  # Reset your environment.
            self._current_step = 0
            return self.observation_space.sample(), {}

        def step(self, action):  # Run a step with an action.
            self._current_step += 1
            observation = self.observation_space.sample()
            rewards = np.random.random()
            terminated = False
            truncated = False if self._current_step < self.max_episode_steps else True
            info = {}
            return observation, rewards, terminated, truncated, info

        def render(self, *args, **kwargs):  # Render your environment and return an image if the render_mode is "rgb_array".
            return np.ones([64, 64, 64])

        def close(self):  # Close your environment.
            return


步骤 2：创建配置文件并读取配置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

接下来，你需要按照 :doc:`“进一步使用” <further_usage>` 中步骤 1 的说明，创建一个 YAML 文件。

下面是一个为 DDPG 算法编写的配置文件示例，文件名为 “ddpg_new_env.yaml”。

.. code-block:: python

    dl_toolbox: "torch"  # The deep learning toolbox. Choices: "torch", "mindspore", "tensorlayer"
    project_name: "XuanCe_Benchmark"
    logger: "tensorboard"  # Choices: tensorboard, wandb.
    wandb_user_name: "your_user_name"
    render: True
    render_mode: 'rgb_array' # Choices: 'human', 'rgb_array'.
    fps: 50
    test_mode: False
    device: "cpu"
    distributed_training: False
    master_port: '12355'

    agent: "DDPG"
    env_name: "MyNewEnv"
    env_id: "new-v1"
    env_seed: 1
    vectorize: "DummyVecEnv"
    policy: "DDPG_Policy"
    representation: "Basic_Identical"
    learner: "DDPG_Learner"
    runner: "DRL"

    representation_hidden_size:  # If you choose Basic_Identical representation, then ignore this value
    actor_hidden_size: [400, 300]
    critic_hidden_size: [400, 300]
    activation: "leaky_relu"
    activation_action: 'tanh'

    seed: 19089
    parallels: 4  # number of environments
    buffer_size: 200000  # replay buffer size
    batch_size: 100
    learning_rate_actor: 0.001
    learning_rate_critic: 0.001
    gamma: 0.99
    tau: 0.005

    start_noise: 0.5
    end_noise: 0.1
    training_frequency: 1
    running_steps: 100000
    start_training: 1000

    use_grad_clip: False  # gradient normalization
    grad_clip_norm: 0.5
    use_obsnorm: False
    use_rewnorm: False
    obsnorm_range: 5
    rewnorm_range: 5

    test_steps: 10000
    eval_interval: 5000
    test_episode: 5

    log_dir: "./logs/ddpg/"
    model_dir: "./models/ddpg/"

然后，读取该配置文件:

.. code-block:: python

    import argparse
    from xuance.common import get_configs
    configs_dict = get_configs(file_dir="ddpg_new_env.yaml")
    configs = argparse.Namespace(**configs_dict)


步骤 3：将环境添加到注册表中
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在定义了新的环境类之后，你需要将其添加到 ``REGISTRY_ENV`` 中。

.. code-block:: python

    from xuance.environment import REGISTRY_ENV
    REGISTRY_ENV[configs.env_name] = MyNewEnv


步骤 4：创建你的环境并在 XuanCe 中运行
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

现在，你可以创建自己的环境，并直接使用 XuanCe 的算法运行它。

以下是使用 DDPG 算法的示例：

.. code-block:: python

    from xuance.environment import make_envs
    from xuance.torch.agents import DDPG_Agent

    envs = make_envs(configs)  # Make parallel environments.
    Agent = DDPG_Agent(config=configs, envs=envs)  # Create a DDPG agent from XuanCe.
    Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
    Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
    Agent.finish()  # Finish the training.


完整代码
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

上述步骤的完整代码可在以下链接查看： `https://github.com/agi-brain/xuance/blob/master/examples/new_environments/ddpg_new_env.py <https://github.com/agi-brain/xuance/blob/master/examples/new_environments/ddpg_new_env.py>`_
