安装教程
===========================

玄策算法库能够在Linux，Windows，MacOS等操作系统上运行，并且易于安装。

安装玄策之前，需首先安装 Anaconda_ 来配置Python运行环境。

安装完 Anaconda 后，可通过以下两种方法安装玄策。

.. raw:: html

   <br><hr>

方法一：通过PyPI安装
---------------------------

**步骤 1**: 打开终端窗口程序，创建新的 conda 环境（Python版本推荐3.7及以上），名称自定义（这里命名为'xpolicy'）:

.. code-block:: console

    conda create -n xpolicy python=3.7

**步骤 2**: conda 环境创建成功后，激活该环境:

.. code-block:: console
    
    conda activate xpolicy

**步骤 3**: 安装玄策:

.. code-block:: console
    
    pip install xuanpolicy

由于没有指定特定的深度学习框架，上面的安装指令没有安装深度学习工具包。用户可通过以下指令按需求安装：

仅安装 PyTorch_ 版玄策:

.. code-block:: console
    
    pip install xuanpolicy[torch]

仅安装 TensorFlow2_ 版玄策:

.. code-block:: console
    
    pip install xuanpolicy[tensorflow]

仅安装 MindSpore_ 版玄策:

.. code-block:: console
    
    pip install xuanpolicy[mindspore]

安装同时支持 PyTorch_，TensorFlow2_，和 MindSpore_ 三种深度学习框架的玄策：

.. code-block:: console

    pip install xuanpolicy[all]


.. raw:: html

   <br><hr>

方法二：本地安装
---------------------------

（注意：步骤1、步骤2和方法一中的相同）

**步骤 1**: 打开终端窗口程序，创建新的 conda 环境（Python版本推荐3.7及以上），名称自定义（这里命名为'xpolicy'）:

.. code-block:: console

    conda create -n xpolicy python=3.7

**步骤 2**: conda 环境创建成功后，激活该环境:

.. code-block:: console
    
    conda activate xpolicy

**步骤 3**: 从GitHub下载xuanpolicy源码:

.. code-block:: console
    
    git clone https://github.com/agi-brain/xuanpolicy.git

**步骤 4**: 下载完毕后，进入xuanpolicy主目录:

.. code-block:: console
    
    cd xuanpolicy

**步骤 5**: 完成本地安装:

.. code-block:: console
    
    pip install -e .


提示: 完成以上步骤后，即可运行大部分玄策中的DRL和MARL算法。个别算法需要安装特殊的python依赖库，用户可根据实际的需求来手动安装。 

.. _Anaconda: https://www.anaconda.com/download
.. _PyTorch: https://pytorch.org/get-started/locally/
.. _TensorFlow2: https://www.tensorflow.org/install
.. _MindSpore: https://www.mindspore.cn/install/en

.. raw:: html

   <br><hr>
   
测试是否安装成功
---------------------------

安装完玄策后，在终端输入python进入python运行环境，输入

.. code-block:: python3

    import xuanpolicy

如果没有发出报错或警告，则表示已成功安装玄策，下一步就可以使用了。