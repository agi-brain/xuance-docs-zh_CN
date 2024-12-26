Customized Configurations
--------------------------

用户也可以选择不适用玄策提供的默认参数，或者玄策中不包含用户的任务时，可用同样的方式自定义.yaml参数配置文件。
但是在获取runner的过程中，需指定参数文件的存放位置，示例如下：

.. code-block:: python

    import xuance as xp
    runner = xp.get_runner(method='dqn',
                           env='classic_control',
                           env_id='CartPole-v1',
                           config_path="xxx/xxx.yaml",
                           is_test=False)
    runner.run()
