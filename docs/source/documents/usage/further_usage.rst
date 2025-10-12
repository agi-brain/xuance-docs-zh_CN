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


在该步骤中，调用了“玄策”中的 ``get_arguments()`` 函数。在该函数中，首先根据 ``env`` 和 ``env_id`` 变量组合，从xuance/configs/路径中查询是否有可读取的参数。
如已经有默认的参数，则全部读取。接着继续从 ``config.path`` 路径下索引步骤1中的配置文件，并读取.yaml文件中的所有参数。最后读取 ``parser`` 中的全部参数。
三次读取中，若遇到相同变量名，则以后者参数为准进行更新。最终， ``get_arguments()`` 函数将返回 ``args`` 变量，包含所有参数信息，输入 ``run()`` 函数中。

.. raw:: html

   <br><hr>
   
步骤3：定义run()，创建模型，运行算法
-----------------------------------------------

定义 ``run()`` 函数，输入为步骤2中得到的 ``args`` 变量。在函数中，实现了环境创建，实例化representation、policy、agent等模块，并实现训练。
以下是带注释的run()函数定义示例：

.. code-block:: python

    import os
    from copy import deepcopy
    import numpy as np
    import torch.optim

    from xuance.common import space2shape
    from xuance.environment import make_envs
    from xuance.torch.utils.operations import set_seed
    from xuance.torch.utils import ActivationFunctions

    def run(args):
        agent_name = args.agent  # 获取智能体名称
        set_seed(args.seed)  # 设置随机种子

        # prepare directories for results
        args.model_dir = os.path.join(os.getcwd(), args.model_dir, args.env_id)  # 模型存储/读取路径
        args.log_dir = os.path.join(args.log_dir, args.env_id)  # 日志文件存储路径

        # build environments
        envs = make_envs(args)  # 创建强化学习环境
        args.observation_space = envs.observation_space  # 获取观测空间
        args.action_space = envs.action_space  # 获取动作空间
        n_envs = envs.num_envs  # 获取并行环境个数

        # prepare representation
        from xuance.torch.representations import Basic_MLP  # 导入表征器类
        representation = Basic_MLP(input_shape=space2shape(args.observation_space),
                                hidden_sizes=args.representation_hidden_size,
                                normalize=None,
                                initialize=torch.nn.init.orthogonal_,
                                activation=ActivationFunctions[args.activation],
                                device=args.device)  # 创建MLP表征器

        # prepare policy
        from xuance.torch.policies import Gaussian_AC_Policy  # 导入策略类
        policy = Gaussian_AC_Policy(action_space=args.action_space,
                                    representation=representation,
                                    actor_hidden_size=args.actor_hidden_size,
                                    critic_hidden_size=args.critic_hidden_size,
                                    normalize=None,
                                    initialize=torch.nn.init.orthogonal_,
                                    activation=ActivationFunctions[args.activation],
                                    device=args.device)  # 创建服从高斯分布的随机策略

        # prepare agent
        from xuance.torch.agents import PPOCLIP_Agent, get_total_iters  # 导入智能体类
        optimizer = torch.optim.Adam(policy.parameters(), args.learning_rate, eps=1e-5)  # 创建优化器
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                                                        total_iters=get_total_iters(agent_name, args))  # 创建学习率衰减器
        agent = PPOCLIP_Agent(config=args,
                              envs=envs,
                              policy=policy,
                              optimizer=optimizer,
                              scheduler=lr_scheduler,
                              device=args.device)  # 创建PPO智能体

        # start running
        envs.reset()  # 环境初始化
        if args.benchmark:  # run benchmark
            def env_fn():  # 创建测试环境，用于每个阶段训练结束后，随机初始化测试环境并进行测试
                args_test = deepcopy(args)  # 拷贝原有参数
                args_test.parallels = args_test.test_episode  # 更改并行环境数量为测试回合数
                return make_envs(args_test)  # 返回实例化测试环境

            train_steps = args.running_steps // n_envs  # 获取智能体总的运行步数
            eval_interval = args.eval_interval // n_envs  # 确定每轮训练步数
            test_episode = args.test_episode  # 获取测试回合数
            num_epoch = int(train_steps / eval_interval)  # 确定训练轮数

            test_scores = agent.test(env_fn, test_episode)  # 第0步测试，得到测试结果
            best_scores_info = {"mean": np.mean(test_scores),  # 平均累积回合奖励
                                "std": np.std(test_scores),  # 累积回合奖励方差
                                "step": agent.current_step}  # 当前步数
            for i_epoch in range(num_epoch):  # 开始轮回训练
                print("Epoch: %d/%d:" % (i_epoch, num_epoch))  # 打印第i_epoch轮训练的基本信息
                agent.train(eval_interval)  # 训练eval_interval步
                test_scores = agent.test(env_fn, test_episode)  # 测试test_episode个回合

                if np.mean(test_scores) > best_scores_info["mean"]:  # 若当前测试结果为历史最高，则保存模型
                    best_scores_info = {"mean": np.mean(test_scores),
                                        "std": np.std(test_scores),
                                        "step": agent.current_step}
                    # save best model
                    agent.save_model(model_name="best_model.pth")
            # end benchmarking
            print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))  # 结束benchmark训练，打印最终结果
        else:
            if not args.test:  # train the model without testing
                n_train_steps = args.running_steps // n_envs  # 确定总的运行步数
                agent.train(n_train_steps)  # 直接训练模型
                agent.save_model("final_train_model.pth")  # 保存最终训练结果
                print("Finish training!")  # 结束训练
            else:  # test a trained model
                def env_fn():
                    args_test = deepcopy(args)
                    args_test.parallels = 1
                    return make_envs(args_test)

                agent.render = True
                agent.load_model(agent.model_dir_load, args.seed)  # 加载模型文件
                scores = agent.test(env_fn, args.test_episode)  # 测试模型
                print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
                print("Finish testing.")  # 结束测试

        # the end.
        envs.close()  # 关闭环境
        agent.finish()  # 结束实验


完成以上三个步骤后，可在终端运行 `ppo_mujoco.py` Python文件，训练模型：

.. code-block:: console

    $ python ppo_mujoco.py --method ppo --env mujoco --env-id Ant-v4


该实例的完整代码见如下链接：

`https://github.com/agi-brain/xuance/examples/ppo/ppo_mujoco.py <https://github.com/agi-brain/xuance/examples/ppo/ppo_mujoco.py/>`_

利用多GPU实现分布式训练
--------------------------------------

XuanCe 支持 多 GPU 并行训练，以最大化 GPU 资源利用率，从而实现更高效的深度强化学习（DRL）模型训练。

若要使用多 GPU 进行 DRL 模型训练，需要将参数 distributed_training 设置为 True。
以下是相关参数说明：
- distributed_training（bool）：指定是否启用多 GPU 分布式训练。设置为 True 时开启分布式训练；若为 False，则不启用。
- master_port（int）：当启用分布式训练时，用于定义当前实验的主端口号。