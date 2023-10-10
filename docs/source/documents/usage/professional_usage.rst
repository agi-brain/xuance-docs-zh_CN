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

定义 ``run()`` 函数，输入为步骤2中得到的 ``args``变量。在函数中，实现了环境创建，实例化representation、policy、agent等模块，并实现训练。
以下是带注释的run()函数定义示例：

::

    import os
    from copy import deepcopy
    import numpy as np
    import torch.optim

    from xuanpolicy.common import space2shape
    from xuanpolicy.environment import make_envs
    from xuanpolicy.torch.utils.operations import set_seed
    from xuanpolicy.torch.utils import ActivationFunctions

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
        from xuanpolicy.torch.representations import Basic_MLP  # 导入表征器类
        representation = Basic_MLP(input_shape=space2shape(args.observation_space),
                                hidden_sizes=args.representation_hidden_size,
                                normalize=None,
                                initialize=torch.nn.init.orthogonal_,
                                activation=ActivationFunctions[args.activation],
                                device=args.device)  # 创建MLP表征器

        # prepare policy
        from xuanpolicy.torch.policies import Gaussian_AC_Policy  # 导入策略类
        policy = Gaussian_AC_Policy(action_space=args.action_space,
                                    representation=representation,
                                    actor_hidden_size=args.actor_hidden_size,
                                    critic_hidden_size=args.critic_hidden_size,
                                    normalize=None,
                                    initialize=torch.nn.init.orthogonal_,
                                    activation=ActivationFunctions[args.activation],
                                    device=args.device)  # 创建服从高斯分布的随机策略

        # prepare agent
        from xuanpolicy.torch.agents import PPOCLIP_Agent, get_total_iters  # 导入智能体类
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


该部分完整代码见如下链接：

`https://github.com/agi-brain/xuanpolicy/blob/master/examples/ppo/ppo_mujoco.py <https://github.com/agi-brain/xuanpolicy/blob/master/examples/ppo/ppo_mujoco.py/>`_