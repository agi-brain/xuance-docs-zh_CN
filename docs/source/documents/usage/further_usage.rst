进一步使用
================================

上一页是通过调用runner来直接运行算法。为了更好地帮助用户理解“玄策”的内部实现流程，
从而便于进一步做算法开发和实现用户自己的强化学习任务，下面以PPO算法训练MuJoCo环境任务为例，
详细介绍如何从底层地调用API实现强化学习模型训练。

.. raw:: html

   <a href="https://colab.research.google.com/github/agi-brain/xuance/blob/master/docs/source/notebook-colab/further_usage.ipynb"
      target="_blank"
      rel="noopener noreferrer"
      style="float: left; margin-left: 0px;">
       <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
   </a>
   <br>
   
步骤1：准备YAML文件，配置参数
--------------------------------

创建 `mujoco.yaml` 文件，并指定相关参数，如下所示：

.. code-block:: yaml

    dl_toolbox: "torch"  # The deep learning toolbox. Choices: "torch", "mindspore", "tensorlayer"
    project_name: "XuanCe_Benchmark"
    logger: "tensorboard"  # Choices: tensorboard, wandb.
    wandb_user_name: "your_user_name"  # The username of wandb when the logger is wandb.
    render: False # Whether to render the environment when testing.
    render_mode: 'rgb_array' # Choices: 'human', 'rgb_array'.
    fps: 50  # The frames per second for the rendering videos in log file.
    test_mode: False  # Whether to run in test mode.
    device: "cuda:0"  # Choose an calculating device.
    distributed_training: False  # Whether to use multi-GPU for distributed training.
    master_port: '12355'  # The master port for current experiment when use distributed training.

    agent: "PPO_Clip"  # The agent name.
    env_name: "MuJoCo"  # The environment device.
    env_id: "Ant-v4"  # The environment id.
    env_seed: 1
    vectorize: "DummyVecEnv"  # The vecrized method to create n parallel environments. Choices: DummyVecEnv, or SubprocVecEnv.
    learner: "PPOCLIP_Learner"
    policy: "Gaussian_AC"  # choice: Gaussian_AC for continuous actions, Categorical_AC for discrete actions.
    representation: "Basic_MLP"  # The representation name.

    representation_hidden_size: [256,]  # The size of hidden layers for representation network.
    actor_hidden_size: [256,]  # The size of hidden layers for actor network.
    critic_hidden_size: [256,]  # The size of hidden layers for critic network.
    activation: "leaky_relu"  # The activation function for each hidden layer.
    activation_action: 'tanh'  # The activation function for the last layer of actor network.

    seed: 79811  # The random seed.
    parallels: 16  # The number of environments to run in parallel.
    running_steps: 1000000  # The total running steps for all environments.
    horizon_size: 256  # the horizon size for an environment, buffer_size = horizon_size * parallels.
    n_epochs: 16  # The number of training epochs.
    n_minibatch: 8  # The number of minibatch for each training epoch. batch_size = buffer_size // n_minibatch.
    learning_rate: 0.0004  # The learning rate.

    vf_coef: 0.25  # Coefficient factor for critic loss.
    ent_coef: 0.0  # Coefficient factor for entropy loss.
    target_kl: 0.25  # For PPO_KL learner.
    kl_coef: 1.0  # For PPO_KL learner.
    clip_range: 0.2  # The clip range for ratio in PPO_Clip learner.
    gamma: 0.99  # Discount factor.
    use_gae: True  # Use GAE trick.
    gae_lambda: 0.95  # The GAE lambda.
    use_advnorm: True  # Whether to use advantage normalization.

    use_grad_clip: True  # Whether to clip the gradient during training.
    clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
    grad_clip_norm: 0.5  # The max norm of the gradient.
    use_actions_mask: False  # Whether to use action mask values.
    use_obsnorm: True  # Whether to use observation normalization.
    use_rewnorm: True  # Whether to use reward normalization.
    obsnorm_range: 5  # The range of observation if use observation normalization.
    rewnorm_range: 5  # The range of reward if use reward normalization.

    test_steps: 10000  # The total steps for testing.
    eval_interval: 5000  # The evaluate interval when use benchmark method.
    test_episode: 5  # The test episodes.
    log_dir: "./logs/ppo/"  # The main directory of log files.
    model_dir: "./models/ppo/"  # The main directory of model files.

   
步骤2：读取参数
-----------------------------------

该部分主要包括参数读取、环境创建、模型构建以及模型训练等内容。
首先，创建一个名为 ppo_mujoco.py 的文件。
代码的编写过程可以分为以下几个步骤：

**Step 2.0 导入必要工具**

.. code-block:: python

    import argparse
    import numpy as np
    from copy import deepcopy
    from xuance.common import get_configs, recursive_dict_update
    from xuance.environment import make_envs
    from xuance.torch.utils.operations import set_seed
    from xuance.torch.agents import PPOCLIP_Agent

**步骤2.1 解析终端命令参数**

定义如下函数 ``parse_args()``，利用Python包 `argparser` 读取终端指令，获取指令参数。

.. code-block:: python

    import argparse

    def parse_args():
        parser = argparse.ArgumentParser("Example of XuanCe: PPO for MuJoCo.")
        parser.add_argument("--env-id", type=str, default="InvertedPendulum-v4")
        parser.add_argument("--test", type=int, default=0)
        parser.add_argument("--benchmark", type=int, default=1)

        return parser.parse_args()


**步骤2.2 读取参数**

首先通过调用步骤2.1中的 ``parse_args()`` 函数读取终端指令参数，然后获取步骤1中的配置参数。

.. code-block:: python

    if __name__ == "__main__":
        parser = parse_args()
        configs_dict = get_configs(file_dir="ppo_configs/ppo_mujoco_config.yaml")
        configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
        configs = argparse.Namespace(**configs_dict)


在该步骤中，调用了“玄策”中的 ``get_configs()`` 方法。该方法可以从指定目录读取配置文件，并返回一个字典变量。
随后，调用“玄策”中的``recursive_dict_update``方法。该方法可以根据 ``parser`` 变量中的内容，对 ``.yaml`` 文件中的配置进行更新。

最后，将该字典变量转换为 ``Namespace`` 类型。

.. raw:: html

   <br><hr>
   
步骤3：创建环境，PPO Agent，运行算法
-----------------------------------------------

.. code-block:: python

    import argparse
    import numpy as np
    from copy import deepcopy
    from xuance.common import get_configs, recursive_dict_update
    from xuance.environment import make_envs
    from xuance.torch.utils.operations import set_seed
    from xuance.torch.agents import PPOCLIP_Agent


    def parse_args():
        parser = argparse.ArgumentParser("Example of XuanCe: PPO for MuJoCo.")
        parser.add_argument("--env-id", type=str, default="InvertedPendulum-v4")
        parser.add_argument("--test", type=int, default=0)
        parser.add_argument("--benchmark", type=int, default=1)

        return parser.parse_args()


    if __name__ == "__main__":
        parser = parse_args()
        configs_dict = get_configs(file_dir="ppo_configs/ppo_mujoco.yaml")
        configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
        configs = argparse.Namespace(**configs_dict)

        set_seed(configs.seed)  # Set the random seed.
        envs = make_envs(configs)  # Make the environment.
        Agent = PPOCLIP_Agent(config=configs, envs=envs)  # Create the PPO agent.

        train_information = {"Deep learning toolbox": configs.dl_toolbox,
                             "Calculating device": configs.device,
                             "Algorithm": configs.agent,
                             "Environment": configs.env_name,
                             "Scenario": configs.env_id}
        for k, v in train_information.items():  # Print the training information.
            print(f"{k}: {v}")

        if configs.benchmark:
            def env_fn():  # Define an environment function for test algo.
                configs_test = deepcopy(configs)
                configs_test.parallels = configs_test.test_episode
                return make_envs(configs_test)

            train_steps = configs.running_steps // configs.parallels
            eval_interval = configs.eval_interval // configs.parallels
            test_episode = configs.test_episode
            num_epoch = int(train_steps / eval_interval)

            test_scores = Agent.test(test_episodes=test_episode, test_envs=test_envs, close_envs=False)
            Agent.save_model(model_name="best_model.pth")
            best_scores_info = {"mean": np.mean(test_scores),
                                "std": np.std(test_scores),
                                "step": Agent.current_step}
            for i_epoch in range(num_epoch):
                print("Epoch: %d/%d:" % (i_epoch, num_epoch))
                Agent.train(eval_interval)
                test_scores = Agent.test(test_episodes=test_episode, test_envs=test_envs, close_envs=False)

                if np.mean(test_scores) > best_scores_info["mean"]:
                    best_scores_info = {"mean": np.mean(test_scores),
                                        "std": np.std(test_scores),
                                        "step": Agent.current_step}
                    # save best model
                    Agent.save_model(model_name="best_model.pth")
            # end benchmarking
            test_envs.close()
            print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
        else:
            if configs.test:
                configs.parallels = configs.test_episode
                test_envs = make_envs(configs)
                Agent.load_model(path=Agent.model_dir_load)
                scores = Agent.test(test_episodes=configs.test_episode, test_envs=test_envs, close_envs=True)
                print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
                print("Finish testing.")
            else:
                Agent.train(configs.running_steps // configs.parallels)
                Agent.save_model("final_train_model.pth")
                print("Finish training!")

        Agent.finish()


完成以上三个步骤后，可在终端运行 `ppo_mujoco.py` Python文件，训练模型：

.. code-block:: console

    $ python ppo_mujoco.py --env-id Ant-v4


该实例的完整代码见如下链接：

`https://github.com/agi-brain/xuance/examples/ppo/ppo_mujoco.py <https://github.com/agi-brain/xuance/examples/ppo/ppo_mujoco.py/>`_

利用多GPU实现分布式训练
--------------------------------------

XuanCe 支持 多 GPU 并行训练，以最大化 GPU 资源利用率，从而实现更高效的深度强化学习（DRL）模型训练。

若要使用多 GPU 进行 DRL 模型训练，需要将参数 distributed_training 设置为 True。
以下是相关参数说明：
- distributed_training（bool）：指定是否启用多 GPU 分布式训练。设置为 True 时开启分布式训练；若为 False，则不启用。
- master_port（int）：当启用分布式训练时，用于定义当前实验的主端口号。