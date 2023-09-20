参数配置
======================

玄策的算法参数均通过YAML文件配置，存于xuanpolicy/configs文件夹中。

为了便于训练和调参，参数配置分为基础参数配置和算法参数配置。

基础参数配置
--------------------------
基础参数配置存于xuanpolicy/config/basic.yaml文件中，各参数名称、含义、取值范围如下表所示：

::

    dl_toolbox: "torch"  # The deep learning toolbox. Choices: "torch", "mindspore", "tensorlayer"

    runner: "DRL"  # Choices: "DRL", "MARL", "Mean_Field_MARL"
    project_name: "XuanPolicy_Benchmark"
    logger: "tensorboard"  # Choices: tensorboard, wandb.
    wandb_user_name: "papers_liu"

    parallels: 10
    seed: 2910
    vectorize: "Dummy_Gym"  # choices: [Dummy_Gym, Dummy_Pettingzoo, Dummy_Atari, Subproc, NOREQUIRED]
    render: True
    render_mode: 'rgb_array' # Choices: 'human', 'rgb_array'.
    test_mode: False
    test_steps: 2000

    device: "cuda:0"

    # PyTorch: "cpu", "cuda:0";
    # TensorFlow: "cpu"/"CPU", "gpu"/"GPU";
    # MindSpore: "CPU", "GPU", "Ascend", "Davinci"


算法参数配置
--------------------------

::

    agent: "DQN"
    vectorize: "Dummy_Atari"
    env_name: "Atari"
    env_id: "ALE/Breakout-v5"
    obs_type: "grayscale"  # choice for Atari env: ram, rgb, grayscale
    img_size: [84, 84]  # default is 210 x 160 in gym[Atari]
    num_stack: 4  # frame stack trick
    frame_skip: 4  # frame skip trick
    noop_max: 30  # Do no-op action for a number of steps in [1, noop_max].
    policy: "Basic_Q_network"
    representation: "Basic_CNN"

    # the following three arguments are for "Basic_CNN" representation.
    filters: [32, 64, 64]  #  [16, 16, 32, 32]
    kernels: [8, 4, 3]  # [8, 6, 4, 4]
    strides: [4, 2, 1]  # [2, 2, 2, 2]

    q_hidden_size: [512, ]
    activation: "ReLU"

    seed: 1069
    parallels: 5
    n_size: 100000
    batch_size: 32  # 64
    learning_rate: 0.0001
    gamma: 0.99

    start_greedy: 0.5
    end_greedy: 0.05
    decay_step_greedy: 1000000  # 1M
    sync_frequency: 500
    training_frequency: 1
    running_steps: 50000000  # 50M
    start_training: 10000

    use_obsnorm: False
    use_rewnorm: False
    obsnorm_range: 5
    rewnorm_range: 5

    test_steps: 10000
    eval_interval: 500000
    test_episode: 3
    log_dir: "./logs/dqn/"
    model_dir: "./models/dqn/"
