Custom Configurations
--------------------------

用户也可以选择不使用 XuanCe 提供的默认参数配置。若 XuanCe 未包含用户所需的特定任务，用户可以以相同的方式自定义自己的 .yaml 参数配置文件。

然而，在获取 runner 的过程中，必须指定参数文件的存储路径，示例如下：

.. code-block:: python

    import xuance as xp
    runner = xp.get_runner(method='dqn',
                           env='classic_control',
                           env_id='CartPole-v1',
                           config_path="xxx/xxx.yaml",
                           is_test=False)
    runner.run()
