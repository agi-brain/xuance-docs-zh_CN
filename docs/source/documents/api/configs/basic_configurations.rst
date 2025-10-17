Basic Configurations
--------------------------

基本参数配置保存在 “xuance/config/basic.yaml” 文件中，示例如下：


.. code-block:: yaml

    dl_toolbox: "torch"  # 选择深度学习框架: "torch", "mindspore", "tensorflow"

    project_name: "XuanCe_Benchmark"
    logger: "tensorboard"  # 选项: "tensorboard", "wandb".
    wandb_user_name: "your_user_name"

    parallels: 10
    seed: 2910
    render: True
    render_mode: 'rgb_array' # Choices: 'human', 'rgb_array'.
    test_mode: False
    test_steps: 2000

    device: "cpu"


需要注意的是，`basic.yaml` 文件中 `device` 变量的取值会根据所使用的深度学习框架而有所不同，具体说明如下：


| - PyTorch: "cpu", "cuda:0";
| - TensorFlow: "cpu"/"CPU", "gpu"/"GPU";
| - MindSpore: "CPU", "GPU", "Ascend", "Davinci".
