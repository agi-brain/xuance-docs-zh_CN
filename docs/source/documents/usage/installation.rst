安装教程
===========================

玄策算法库能够在Linux，Windows，MacOS等操作系统上运行，并且易于安装。

安装玄策之前，需首先安装 Anaconda_ 来配置Python运行环境。

安装完 Anaconda 后，可通过以下两种方法安装玄策。

.. note::

    XuanCe 可以在 macOS 系统上安装，并兼容 Intel 芯片和 Apple M 系列（如 M1、M2、M3等）处理器。

安装玄策
---------------------------

**步骤 1**: 创建并激活一个新的 Conda 环境（建议使用 Python 3.8 或更高版本）。

.. code-block:: bash

    conda create -n xuance_env python=3.8 && conda activate xuance_env

**步骤 2**: 安装 ``xuance``.

.. tabs::

    .. tab:: |_3| |torch| |_3| (默认)

        .. code-block:: bash

            pip install xuance

    .. tab:: |_3| |tensorflow| |_3|

        .. code-block:: bash

            pip install xuance[tensorflow]

    .. tab:: |_3| |mindspore| |_3|

        .. code-block:: bash

            pip install xuance[mindspore]

    .. tab:: 全部

        .. code-block:: bash

            pip install xuance[all]

或者，你也可以直接从其 `GitHub <https://github.com/agi-brain/xuance.git/>`_ 仓库 安装 ``xuance``。

.. tabs::

    .. tab:: |_3| |torch| |_3| (默认)

        .. code-block:: bash

            git clone https://github.com/agi-brain/xuance.git
            cd xuance
            pip install -e .[torch]

    .. tab:: |_3| |tensorflow| |_3|

        .. code-block:: bash

            git clone https://github.com/agi-brain/xuance.git
            cd xuance
            pip install -e .[tensorflow]

    .. tab:: |_3| |mindspore| |_3|

        .. code-block:: bash

            git clone https://github.com/agi-brain/xuance.git
            cd xuance
            pip install -e .[mindspore]

    .. tab:: 全部

        .. code-block:: bash

            git clone https://github.com/agi-brain/xuance.git
            cd xuance
            pip install -e .[all]

.. attention::

    为后续使用，你需要手动安装一些额外的依赖包。
    详见 `Install external dependencies <#id1>`_

.. error::

    在安装 XuanCe 的过程中，你可能会遇到以下错误：

    .. code-block:: bash

        Error: Failed to building wheel for mpi4py
        Failed to build mpi4py
        ERROR: Could not build wheels for mpi4py, which is required to install pyproject.toml-based projects

    **Solution 1**: 你可以通过手动安装 mpi4py 来解决该问题，命令如下：

    .. code-block:: bash

        conda install mpi4py

    **Solution 2**: 如果上述方法仍未解决问题，你可以通过以下命令安装 ``gcc_linux-64``：

    .. code-block:: bash

        conda install gcc_linux-64

    然后，再次使用 pip 输入以下命令以重新安装 mpi4py：

    .. code-block:: bash

        pip install mpi4py

.. note::

    该问题仅影响 v1.4.0 之前的 XuanCe 版本。
    从 v1.4.0 开始，mpi4py 不再是 XuanCe 的必需依赖。

    如果你正在使用较旧版本的 XuanCe 并遇到该错误，
    可以按照上述方法进行解决。

.. tip::

    如果你的 IP 地址位于中国大陆，可以使用镜像源来加快安装速度。
    例如，你可以选择以下任意一条命令来完成安装：

    .. code-block:: bash

        pip install xuance -i https://pypi.tuna.tsinghua.edu.cn/simple
        pip install xuance -i https://pypi.mirrors.ustc.edu.cn/simple
        pip install xuance -i http://mirrors.aliyun.com/pypi/simple/
        pip install xuance -i http://pypi.douban.com/simple/

.. _Anaconda: https://www.anaconda.com/download
.. _PyTorch: https://pytorch.org/get-started/locally/
.. _TensorFlow2: https://www.tensorflow.org/install
.. _MindSpore: https://www.mindspore.cn/install/en
   
测试是否安装成功
---------------------------

安装完成后，你可以在终端中输入 python 进入 Python 运行环境。
接着，输入以下命令来测试 XuanCe 是否已成功安装：

.. code-block:: python

    import xuance


.. error::

    如果你在 Windows 操作系统 下导入 XuanCe，可能会遇到如下错误：

    .. code-block:: bash

        ...
        from mpi4py import MPI
        ImportError: DLL load failed: The specified module could not be found.

    你可以通过一下几个步骤解决该问题：

    **Step 1**: 下载 Microsoft MPI v10.0（`Microsoft 官方下载中心 <https://www.microsoft.com/en-us/download/details.aspx?id=57467>`_）.

    **Step 2**: 记得同时选择 "msmpisetup.exe" 和 "msmpisdk.msi" 选项, 然后点击 "Download" 按钮并安装 ".exe" 文件.

    **Step 3**: 重新安装 mpi4py:

    .. code-block:: bash

        pip uninstall mpi4py
        pip install mpi4py


如果没有出现任何错误或警告信息，说明 XuanCe 已成功安装。
你可以继续进行下一步，开始使用它。 (`点击此处跳转到下一页 <basic_usage.html>`_)

.. raw:: html

    <br><hr>

安装外部依赖
-------------------------------

部分依赖项未包含在 XuanCe 的安装过程中。
你可以根据需要，手动安装下方列出的外部依赖包。

Box2D
^^^^^^^^

`Box2D 环境 <../api/environments/single_agent_env/gym.html#box2d>`_ 是基于 `box2d <https://box2d.org/>`_ 构建的，用于物理控制任务。
它包含三种不同的任务：Bipedal Walker（双足行走者）、Car Racing（赛车） 和 Lunar Lander（月球着陆器）。
如果你想尝试这些任务，可以通过以下命令进行安装。

.. tabs::

    .. tab:: 通过 PyPI 安装

        .. code-block:: bash

            pip install swig==4.3.0
            pip install gymnasium[box2d]==0.28.1

    .. tab:: 通过 XuanCe 安装

        .. code-block:: bash

            pip install xuance[box2d]

MuJoCo
^^^^^^^^

`MuJoCo 环境 <../api/environments/single_agent_env/gym.html#mujoco>`_ 是一个物理引擎，用于促进机器人学、生物力学、图形与动画等领域的研究与开发，
以及其他需要快速且精确模拟的应用场景。

**Step 1: 安装 MuJoCo**

* 下载适用于 `Linux <https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz>`_ 或 `OSX <https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz>`_ 的 MuJoCo 2.1 二进制文件.
* 解压所下载的 ``mujoco210`` 文件夹并复制到 ``~/.mujoco/mujoco210`` 文件夹中.

**Step 2: 安装 mujoco-py**

.. tabs::

    .. tab:: 通过 PyPI 安装

        .. code-block:: bash

            pip install gymnasium[mujoco]

    .. tab:: 通过 XuanCe 安装

        .. code-block:: bash

            pip install xuance[mujoco]

Atari
^^^^^^^^

`Atari 环境 <../api/environments/single_agent_env/gym.html#atari>`_ 是通过
`Arcade Learning Environment (ALE) <https://www.jair.org/index.php/jair/article/view/10819>`_ 模拟实现的，
其中包含 62 种不同的任务。

.. tabs::

    .. tab:: 通过 PyPI 安装

        .. code-block:: bash

            pip install gymnasium[accept-rom-license]==0.28.1
            pip install gymnasium[atari]==0.28.1
            pip install ale-py==0.8.1

    .. tab:: 通过 XuanCe 安装

        .. code-block:: bash

            pip install xuance[atari]

MiniGrid
^^^^^^^^^

`MiniGrid 环境 <../api/environments/single_agent_env/minigrid.html>`_ 是一个轻量级的基于网格的环境，专为深度强化学习研究设计。
它具有高度的可定制性，支持多种任务与挑战，可用于训练在部分可观测、稀疏奖励以及符号输入条件下的智能体。

.. tabs::

    .. tab:: 通过 PyPI 安装

        .. code-block::

            pip install minigrid

    .. tab:: 通过 GitHub 仓库安装

        .. code-block::

            git clone https://github.com/Farama-Foundation/Minigrid.git
            cd Minigrid
            pip install -e .

    .. tab:: 通过 XuanCe 安装

        .. code-block::

            pip install xuance[minigrid]


MetaDrive
^^^^^^^^^^^

`MetaDrive <../api/environments/single_agent_env/metadrive.html>`_ 是一个自动驾驶模拟器，支持基于不同的道路地图和交通设置生成无限场景，
可用于研究并验证强化学习的可泛化性。

.. tabs::

    .. tab:: 通过 PyPI 安装

        .. code-block::

            pip install metadrive

    .. tab:: 通过 GitHub 仓库安装

        .. code-block::

            git clone https://github.com/metadriverse/metadrive.git
            cd metadrive
            pip install -e .

    .. tab:: 通过 XuanCe 安装

        .. code-block::

            pip install xuance[metadrive]


StarCraft2
^^^^^^^^^^^^

`StarCraft multi-agent challenge (SMAC) <../api/environments/multi_agent_env/smac.html>`_ 是由 `WhiRL's <https://whirl.cs.ox.ac.uk/>`_ 团队开发的多智能体协作强化学习（MARL）研究环境。
SMAC 基于 《星际争霸 II》（由暴雪娱乐公司开发的实时战略游戏）构建，作为其底层仿真环境。

.. note::

    在安装``smac``包之前，请确保你的 Python 版本为 3.8 或更低；否则，在渲染环境时可能会遇到错误。

**Step 1: 安装 ``smac`` python 包**

您可以直接从GitHub安装 SMAC 软件包:

.. tabs::

    .. tab:: 方法 1

        .. code-block:: bash

            pip install git+https://github.com/oxwhirl/smac.git

    .. tab:: 方法 2

        .. code-block:: bash

            git clone https://github.com/oxwhirl/smac.git
            cd smac/
            pip install -e .


**Step 2: 安装 StarCraft II 游戏引擎**

.. tabs::

    .. tab:: Linux 系统

        请使用 `暴雪（Blizzard）官方仓库 <https://github.com/Blizzard/s2client-proto?tab=readme-ov-file#downloads>`_
        下载**星际争霸 II（StarCraft II）**的 **Linux 版本**。

    .. tab:: Windows/MacOS 系统

        你需要先从 `BATTAL.NET <https://battle.net/>`_
        或 `https://starcraft2.blizzard.com <http://battle.net/sc2/en/legacy-of-the-void/>`_ 安装**星际争霸 II（StarCraft II）**。

.. note::

    你需要设置环境变量**SC2PATH**，并指定**星际争霸 II（StarCraft II）**的正确安装路径。
    默认情况下，游戏路径应为 ~/StarCraftII/ 目录。
    你也可以通过设置环境变量**SC2PATH**来修改该路径。

**Step 3: 配置 SMAC 场景地图**

在你安装好``smac``和**StarCraft II**之后，需要下载 `SMAC 地图（SMAC Maps） <https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip>`_,
并将其解压到目录 ``$SC2PATH/Maps$`` 下。

如果你是通过 git 安装的``smac``，只需将
``smac/env/starcraft2/maps/`` 目录中的``SMAC_Maps``文件夹复制到 ``$SC2PATH/Maps$`` 目录中即可。

Google Research Football
^^^^^^^^^^^^^^^^^^^^^^^^^

`Google Research Football 环境 (GRF) <../api/environments/multi_agent_env/football.html>`_ 是由 Google Brain 团队 开发的多智能体强化学习（MARL）环境。
它专为强化学习研究设计，特别适用于多智能体强化学习场景。

**Step 1: 安装所需的依赖包**

.. tabs::

    .. tab:: Linux系统

        .. code-block:: bash

            sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
            libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
            libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip

            python3 -m pip install --upgrade pip setuptools psutil wheel

    .. tab:: MacOS

        .. code-block:: bash

            brew install git python3 cmake sdl2 sdl2_image sdl2_ttf sdl2_gfx boost boost-python3

            python3 -m pip install --upgrade pip setuptools psutil wheel

    .. tab:: Windows系统

        .. code-block::

            python -m pip install --upgrade pip setuptools psutil wheel

**Step 2: 安装 gfootball 依赖包**

.. tabs::

    .. tab:: 通过 PyPI 安装

        .. code-block:: bash

            python3 -m pip install gfootball

    .. tab:: 通过 GitHub 仓库安装

        .. code-block:: bash

            git clone https://github.com/google-research/football.git
            cd football
            python3 -m pip install .


.. attention::

    包括 ``gfootball`` 在内的所有 Python 依赖包，都应安装在同一个 Conda 环境 中。
    具体步骤可参考：`https://xuance.readthedocs.io/en/latest/documents/usage/installation.html#install-xuance <https://xuance.readthedocs.io/en/latest/documents/usage/installation.html#install-xuance>`_.


Robotic Warehouse
^^^^^^^^^^^^^^^^^^^

`Robotic Warehouse <../api/environments/multi_agent_env/robotic_warehouse.html>`_ 是一个多智能体强化学习（MARL）环境，常用于模拟仓库自动化场景。
它作为研究平台，用于探讨多个智能体（如机器人）之间的协作、竞争及混合交互。
该环境旨在重现现实仓库中的典型任务，例如导航、物品取放、避障以及任务分配等。

.. tabs::

    .. tab:: 通过 PyPI 安装

        .. code-block::

            pip install rware

    .. tab:: 通过 GitHub 仓库安装

        .. code-block::

            git clone git@github.com:uoe-agents/robotic-warehouse.git
            cd robotic-warehouse
            pip install -e .

    .. tab:: 通过 XuanCe 安装

        .. code-block::

            pip install xuance[rware]


gym-pybullet-drones
^^^^^^^^^^^^^^^^^^^^

.. tip::

    在准备该模拟器所需的软件包之前，建议先创建一个包含**Python 3.10**的新的 Conda 环境。

打开终端（Terminal），并输入以下命令，即可创建一个用于 XuanCe 无人机环境 的新的 Conda 环境：

.. code-block:: bash

    conda create -n xuance_drones python=3.10
    conda activate xuance_drones
    pip install xuance  # refer to the installation of XuanCe.

    git clone https://github.com/utiasDSL/gym-pybullet-drones.git
    cd gym-pybullet-drones/
    pip install --upgrade pip
    pip install -e .  # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`

在安装 gym-pybullet-drones 的过程中，你可能会遇到如下错误：

.. error::

    | gym-pybullet-drones 2.0.0 requires numpy<2.0,>1.24, but you have numpy 1.22.4 which is incompatible.
    | gym-pybullet-drones 2.0.0 requires scipy<2.0,>1.10, but you have scipy 1.7.3 which is incompatible.

**解决办法**: 将上述不兼容的软件包升级到兼容版本。

.. code-block:: bash

    pip install numpy==1.24.0
    pip install scipy==1.12.0

DCG 算法依赖 (torch-scatter)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 XuanCe 项目中，DCG 算法依赖于 torch-scatter 库。
在大多数情况下，你可以直接使用以下命令来安装该库：

.. code-block:: bash

    pip install torch-scatter

然而，在某些系统（例如特定的操作系统或硬件环境）上，上述命令可能会导致安装错误。

为确保兼容性，请按照以下步骤正确安装 torch-scatter：

**1. 检查你的 PyTorch 和 CUDA 版本**

使用以下命令查看当前环境中安装的 PyTorch 和 CUDA 版本：

.. code-block:: bash

    python -c "import torch; print(torch.__version__, torch.version.cuda)"

请记下你的 PyTorch 版本（例如：2.0.1）和 CUDA 版本（例如：11.8），因为它们将在选择合适的 torch-scatter 版本时用到。

**2. 参考官方的 torch-scatter 安装指南**

访问 `官方 torch-scatter 安装页面 <https://pypi.org/project/torch-scatter/>`_ (需要联网).
找到与你的 PyTorch 和 CUDA 版本匹配的安装命令。

例如：

- 如果你的 PyTorch 版本是 2.0.1，CUDA 版本是 11.8，则运行以下命令：

.. code-block:: bash

    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

- 如果你使用的是 仅支持 CPU 的 PyTorch 版本，请选择带有 +cpu 的安装链接，例如：

.. code-block:: bash

    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cpu.html

**3. 兼容性问题排查**

如果在安装过程中遇到问题，请确保以下几点：
- PyTorch 已正确安装，并且其版本与你选择的 torch-scatter 轮子文件版本匹配。
- Python 和 pip 的版本是最新的。你可以使用以下命令更新 pip：

.. code-block:: bash

    python -m pip install --upgrade pip

**4. 验证安装是否成功**

安装完成后，可以通过运行以下命令验证 torch-scatter 是否成功安装：

.. code-block:: bash

    python -c "import torch_scatter; print('torch-scatter installed successfully')"
