专业教程
================================

上一页通过调用runner直接运行算法的方式，不能帮助用户很好地理解“玄策”的内部实现流程。
为了便于进一步做算法开发，实现用户自己的强化学习任务，下面以PPO算法训练MuJoCo环境任务为例，
详细介绍如何从底层地调用API实现强化学习模型训练。

步骤1：准备YAML文件，配置参数
--------------------------------

创建 `mujoco.yaml` 文件，并指定相关参数，如下所示：

::

    dl_toolbox: "torch"  # The deep learning toolbox. Choices: "torch", "mindspore", "tensorlayer"
    project_name: "XuanPolicy_Benchmark"
    logger: "tensorboard"  # Choices: tensorboard, wandb.
    wandb_user_name: "your_user_name"
    render: False
    render_mode: 'rgb_array' # Choices: 'human', 'rgb_array'.
    test_mode: False
    device: "cuda:0"

    agent: "PPO_Clip"  # choice: PPO_Clip, PPO_KL
    env_name: "MuJoCo"
    vectorize: "Dummy_Gym"
    runner: "DRL"

    representation_hidden_size: [256,]
    actor_hidden_size: [256,]
    critic_hidden_size: [256,]
    activation: "LeakyReLU"

    seed: 79811
    parallels: 16
    running_steps: 1000000
    n_steps: 256
    n_epoch: 16
    n_minibatch: 8
    learning_rate: 0.0004

    use_grad_clip: True

    vf_coef: 0.25
    ent_coef: 0.0
    target_kl: 0.001  # for PPO_KL agent
    clip_range: 0.2  # for PPO_Clip agent
    clip_grad_norm: 0.5
    gamma: 0.99
    use_gae: True
    gae_lambda: 0.95
    use_advnorm: True

    use_obsnorm: True
    use_rewnorm: True
    obsnorm_range: 5
    rewnorm_range: 5

    test_steps: 10000
    eval_interval: 5000
    test_episode: 5
    log_dir: "./logs/ppo/"
    model_dir: "./models/ppo/"

步骤2：读取参数
-----------------------------------

该部分主要包括参数读取、环境创建、模型创建、模型训练等环节，分为如下步骤：

**步骤2.1 解析终端命令参数**

定义如下函数 ``parse_args()``，利用Python包 `argparser` 读取终端指令，获取指令参数。

::
    import argparser

    def parse_args():
        parser = argparse.ArgumentParser("Example of XuanPolicy.")
        parser.add_argument("--method", type=str, default="ppo")
        parser.add_argument("--env", type=str, default="mujoco")
        parser.add_argument("--env-id", type=str, default="InvertedPendulum-v4")
        parser.add_argument("--test", type=int, default=0)
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--benchmark", type=int, default=1)
        parser.add_argument("--config", type=str, default="./ppo_mujoco_config.yaml")

        return parser.parse_args()

**步骤2.2 读取参数**

首先通过调用步骤2.1中的 ``parse_args()``函数读取终端指令参数，然后获取步骤1中的配置参数。

::
    from xuanpolicy import get_arguments

    if __name__ == "__main__":
    parser = parse_args()
    args = get_arguments(method=parser.method,
                         env=parser.env,
                         env_id=parser.env_id,
                         config_path=parser.config,
                         parser_args=parser)
    run(args)

在该步骤中，调用了“玄策”中的 ``get_arguments()`` 函数。在该函数中，首先根据 ``env`` 和 ``env_id``变量组合，从xuanpolicy/configs/路径中查询是否有可读取的参数。
如已经有默认的参数，则全部读取。接着继续从 ``config.path`` 路径下索引步骤1中的配置文件，并读取.yaml文件中的所有参数。最后读取 ``parser``中的全部参数。
三次读取中，若遇到相同变量名，则以后者参数为准进行更新。最终，``get_arguments()`` 函数将返回 ``args`` 变量，包含所有参数信息，输入 ``run()``函数中。

步骤3：定义run()，创建模型，运行算法
-----------------------------------------------

