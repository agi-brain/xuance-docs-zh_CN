自定义回调函数
---------------------------------

在 XuanCe 中，智能体（agent）支持在训练与测试过程中注入用户自定义的回调函数（callback），以实现更高层次的自定义控制与灵活扩展。

你可以通过继承 BaseCallback 类，重写以下任意方法，然后将自定义回调的实例传入智能体中，即可在特定阶段执行自定义逻辑。

可用的回调钩子（Callback Hooks）:

- ``on_update_start(...)``：在策略更新开始前调用。
- ``on_update_end(...)``：在策略更新完成后调用。
- ``on_train_step(...)``：在每个训练步结束后调用。
- ``on_train_epochs_end(...)``：在每个训练轮次结束后调用（即完成一次数据采样后）。
- ``on_train_episode_info(...)``：在某个环境的一个回合（episode）结束或被截断时调用。
- ``on_train_step_end(...)``：在训练步结束后调用（包括更新、日志记录等操作）。
- ``on_test_step(...)``：在测试循环的每一步执行时调用。
- ``on_test_end(...)``：在测试循环结束时调用。
- ``on_update_agent_wise(...)``：在完成某个智能体策略更新后调用。

示例
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

以下示例展示了如何通过自定义回调，在训练过程中注入钩子函数。
在该示例中，我们将在 TensorBoard 上可视化额外的环境相关信息。

在使用此回调之前，请确保环境的 step() 函数返回的 info 字典中包含键 'info_1' 与 'info_2'。
这些值将通过 SummaryWriter 记录，并可在 TensorBoard 中进行展示。

示例代码如下：

.. code-block:: python

    import os
    from xuance.torch.agents import BaseCallback
    from torch.utils.tensorboard import SummaryWriter

    class MyCallback(BaseCallback):
    "The customized callback."
    def __init__(self, config):
        super(MyCallback, self).__init__()
        log_dir = os.path.join(os.getcwd(), config.log_dir, 'callback_info')
        create_directory(log_dir)
        self.writer = SummaryWriter(log_dir)

    def on_train_episode_info(self, *args, **kwargs):
        "Visualize the additional information about the environment on Tensorboard."
        infos = kwargs['infos']
        env_id = kwargs['env_id']
        step = kwargs['current_step']
        self.writer.add_scalars('environment_information/info_1', {f"env-{env_id}": infos[env_id]["info_1"]}, step)
        self.writer.add_scalars('environment_information/info_2', {f"env-{env_id}": infos[env_id]["info_2"]}, step)

    Agent = DQN_Agent(config=configs, envs=envs, callback=MyCallback(configs))  # Create a DDPG agent with customized callback.

完整代码
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

上述示例的完整代码可在以下链接中查看：`https://github.com/agi-brain/xuance/blob/master/examples/new_environments/dqn_new_env.py <https://github.com/agi-brain/xuance/blob/master/examples/new_environments/dqn_new_env.py>`_
