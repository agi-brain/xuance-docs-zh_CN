Configs
======================

Configs模块用于存放有关算法、环境、系统配置等参数，玄策提供的算法案例，其参数均存于该模块中。

为了便于训练和调参，玄策给出了默认的参数配置。
默认参数配置分为基础参数配置和算法参数配置，二者中的参数名称可重复，
但是算法参数配置的优先级高于基础参数配置，即重复的参数名已算法参数配置文件中的值为主。

针对不同的算法、不同的环境，强化学习示例需要用到的参数种类各不相同，因此需要根据实际需求配置必要的参数。
玄策中的所有算法均有相应的示例，每个示例给出了该算法运行所必须的参数。若用户在实现自己的新任务时需要用到其它参数，
可直接在参数配置文件中添加。

.. toctree::
    :hidden:

    Basic Configurations <configs/basic_configurations>
    Configuration Examples <configs/configuration_examples>
    Customized Configurations <configs/customized_configurations>

* :doc:`Basic Configurations <configs/basic_configurations>`.
* :doc:`Configuration Examples <configs/configuration_examples>`.
* :doc:`Customized Configurations <configs/customized_configurations>`.
