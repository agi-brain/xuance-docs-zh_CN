安装教程
===========================

玄策算法库能够在Linux，Windows，MacOS等操作系统上运行，并且易于安装。

安装玄策之前，需首先安装 Anaconda_ 来配置Python运行环境。

安装完 Anaconda 后，可通过以下两种方法安装玄策。

.. note::

    XuanCe 可以在 macOS 系统上安装，并兼容 Intel 芯片和 Apple M 系列（如 M1、M2、M3等）处理器。

安装玄策
---------------------------

**Step 1**: 创建并激活一个新的 Conda 环境（建议使用 Python 3.8 或更高版本）。

.. code-block:: bash

    conda create -n xuance_env python=3.8 && conda activate xuance_env

**Step 2**: 安装 ``mpi4py`` 依赖.

.. code-block:: bash

    conda install mpi4py

**Step 3**: 安装 ``xuance``.

.. tabs::

    .. tab:: 默认（torch）

        .. code-block:: bash

            pip install xuance

    .. tab:: |_3| |torch| |_3|

        .. code-block:: bash

            pip install xuance[torch]

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

    .. tab:: 默认（torch）

        .. code-block:: bash

            git clone https://github.com/agi-brain/xuance.git
            cd xuance
            pip install -e .

    .. tab:: |_4| |torch| |_4|

        .. code-block:: bash

            git clone https://github.com/agi-brain/xuance.git
            cd xuance
            pip install -e .[torch]

    .. tab:: |tensorflow|

        .. code-block:: bash

            git clone https://github.com/agi-brain/xuance.git
            cd xuance
            pip install -e .[tensorflow]

    .. tab:: |mindspore|

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

`MuJoCo environment <../api/environments/single_agent_env/gym.html#mujoco>`_ is a physics engine for facilitating research and development in robotics, biomechanics, graphics and animation,
and other areas where fast and accurate simulation is needed.

**Step 1: Install MuJoCo**

* Download the MuJoCo version 2.1 binaries for `Linux <https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz>`_ or `OSX <https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz>`_.
* Extract the downloaded ``mujoco210`` directory into ``~/.mujoco/mujoco210``.

**Step 2: Install mujoco-py**

.. tabs::

    .. tab:: From PyPI

        .. code-block:: bash

            pip install gymnasium[mujoco]

    .. tab:: From XuanCe

        .. code-block:: bash

            pip install xuance[mujoco]

Atari
^^^^^^^^

`Atari environment <../api/environments/single_agent_env/gym.html#atari>`_ is simulated via the
`Arcade Learning Environment (ALE) <https://www.jair.org/index.php/jair/article/view/10819>`_,
which contains 62 different tasks.

.. tabs::

    .. tab:: From PyPI

        .. code-block:: bash

            pip install gymnasium[accept-rom-license]==0.28.1
            pip install gymnasium[atari]==0.28.1
            pip install ale-py==0.8.1

    .. tab:: From XuanCe

        .. code-block:: bash

            pip install xuance[atari]

MiniGrid
^^^^^^^^^

`MiniGrid environment <../api/environments/single_agent_env/minigrid.html>`_ is a lightweight, grid-based environment designed for research in DRL.
It is highly customizable, supporting a variety of tasks and challenges for training agents
with partial observability, sparse rewards, and symbolic inputs.

.. tabs::

    .. tab:: From PyPI

        .. code-block::

            pip install minigrid

    .. tab:: From GitHub Repository

        .. code-block::

            git clone https://github.com/Farama-Foundation/Minigrid.git
            cd Minigrid
            pip install -e .

    .. tab:: From XuanCe

        .. code-block::

            pip install xuance[minigrid]


MetaDrive
^^^^^^^^^^^

`MetaDrive <../api/environments/single_agent_env/metadrive.html>`_ is an autonomous driving simulator that supports generating infinite scenes with various road maps and traffic settings for research of generalizable RL.

.. tabs::

    .. tab:: From PyPI

        .. code-block::

            pip install metadrive

    .. tab:: From GitHub Repository

        .. code-block::

            git clone https://github.com/metadriverse/metadrive.git
            cd metadrive
            pip install -e .

    .. tab:: From XuanCe

        .. code-block::

            pip install xuance[metadrive]


StarCraft2
^^^^^^^^^^^^

The `StarCraft multi-agent challenge (SMAC) <../api/environments/multi_agent_env/smac.html>`_ is `WhiRL's <https://whirl.cs.ox.ac.uk/>`_ environment for research of cooperative MARL algorithms.
SMAC uses StarCraft II, a real-time strategy game developed by Blizzard Entertainment, as its underlying environment.

.. note::

    Before installing the ``smac`` package, make sure your Python version is 3.8 or lower; otherwise, you may encounter errors when rendering the environment.

**Step 1: Install the smac python package**

You can install the SMAC package directly from the GitHub:

.. tabs::

    .. tab:: Method 1

        .. code-block:: bash

            pip install git+https://github.com/oxwhirl/smac.git

    .. tab:: Method 2

        .. code-block:: bash

            git clone https://github.com/oxwhirl/smac.git
            cd smac/
            pip install -e .


**Step 2: Install StarCraft II**

.. tabs::

    .. tab:: Linux

        Please use the `Blizzard's repository <https://github.com/Blizzard/s2client-proto?tab=readme-ov-file#downloads>`_
        to download the Linux version of StarCraft II.

    .. tab:: Windows/MacOS

        You need to first install StarCraft II from `BATTAL.NET <https://battle.net/>`_
        or `https://starcraft2.blizzard.com <http://battle.net/sc2/en/legacy-of-the-void/>`_.

.. note::

    You would need to set the SC2PATH environment variable with the correct location of the game.
    By default, the game is expected to be in ~/StarCraftII/ directory.
    This can be changed by setting the environment variable SC2PATH.

**Step 3: SMAC Maps**

Once you have installed ``smac`` and StarCraft II, you need to download the
`SMAC Maps <https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip>`_,
and extract it to the ``$SC2PATH/Maps$`` directory.
If you installed ``smac`` via git, simply copy the ``SMAC_Maps`` directory
from ``smac/env/starcraft2/maps/`` into ``$SC2PATH/Maps`` directory.

Google Research Football
^^^^^^^^^^^^^^^^^^^^^^^^^

`Google Research Football Environment (GRF) <../api/environments/multi_agent_env/football.html>`_ is an MARL environment developed by the Google Brain team.
It is specifically designed for RL research, particularly for MARL scenarios.

**Step 1: Install required packages**

.. tabs::

    .. tab:: Linux

        .. code-block:: bash

            sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
            libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
            libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip

            python3 -m pip install --upgrade pip setuptools psutil wheel

    .. tab:: MacOS

        .. code-block:: bash

            brew install git python3 cmake sdl2 sdl2_image sdl2_ttf sdl2_gfx boost boost-python3

            python3 -m pip install --upgrade pip setuptools psutil wheel

    .. tab:: Windows

        .. code-block::

            python -m pip install --upgrade pip setuptools psutil wheel

**Step 2: Install gfootball**

.. tabs::

    .. tab:: From PyPI

        .. code-block:: bash

            python3 -m pip install gfootball

    .. tab:: From GitHub repository

        .. code-block:: bash

            git clone https://github.com/google-research/football.git
            cd football
            python3 -m pip install .


.. attention::

    All python packages including ``gfootball`` environment should be installed in a same conda environment.
    See `https://xuance.readthedocs.io/en/latest/documents/usage/installation.html#install-xuance <https://xuance.readthedocs.io/en/latest/documents/usage/installation.html#install-xuance>`_.


Robotic Warehouse
^^^^^^^^^^^^^^^^^^^

`Robotic Warehouse <../api/environments/multi_agent_env/robotic_warehouse.html>`_ is an MARL environment often used to simulate warehouse automation scenarios.
It serves as a testbed for studying cooperative, competitive, and mixed interaction among multiple agents, such as robots.
The environment is designed to model tasks commonly found in real-world warehouses,
such as navigation, item retrieval, obstacle avoidance, and task allocation.

.. tabs::

    .. tab:: From PyPI

        .. code-block::

            pip install rware

    .. tab:: From GitHub Repository

        .. code-block::

            git clone git@github.com:uoe-agents/robotic-warehouse.git
            cd robotic-warehouse
            pip install -e .

    .. tab:: From XuanCe

        .. code-block::

            pip install xuance[rware]


gym-pybullet-drones
^^^^^^^^^^^^^^^^^^^^

.. tip::

    Before preparing the software packages for this simulator, it is recommended to create a new conda environment with **Python 3.10**.

Open terminal and type the following commands, then a new conda environment for xuance with drones could be built:

.. code-block:: bash

    conda create -n xuance_drones python=3.10
    conda activate xuance_drones
    pip install xuance  # refer to the installation of XuanCe.

    git clone https://github.com/utiasDSL/gym-pybullet-drones.git
    cd gym-pybullet-drones/
    pip install --upgrade pip
    pip install -e .  # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`

During the installation of gym-pybullet-drones, you might encounter the errors like:

.. error::

    | gym-pybullet-drones 2.0.0 requires numpy<2.0,>1.24, but you have numpy 1.22.4 which is incompatible.
    | gym-pybullet-drones 2.0.0 requires scipy<2.0,>1.10, but you have scipy 1.7.3 which is incompatible.

**Solution**: Upgrade the above incompatible packages.

.. code-block:: bash

    pip install numpy==1.24.0
    pip install scipy==1.12.0

DCG algorithm dependency (torch-scatter)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The DCG algorithm in the XuanCe project relies on the torch-scatter library.
In most cases, you can install it directly using the following command:

.. code-block:: bash

    pip install torch-scatter

However, on certain systems (e.g., specific operating systems or hardware environments),
this command may result in installation errors.
To ensure compatibility, follow the steps below to correctly install torch-scatter:

**1. Check Your PyTorch and CUDA Versions**

Use the following command to check the installed version of PyTorch and CUDA in your environment:

.. code-block:: bash

    python -c "import torch; print(torch.__version__, torch.version.cuda)"

Take note of the PyTorch version (e.g., 2.0.1) and the CUDA version (e.g., 11.8) as they will be needed to select the appropriate version of torch-scatter.

**2. Refer to the Official torch-scatter Installation Guide**

Visit the `official torch-scatter installation page <https://pypi.org/project/torch-scatter/>`_ (internet connection required).
Find the installation command that matches your PyTorch and CUDA versions. For example:

- If your PyTorch version is 2.0.1 and CUDA version is 11.8, run:

.. code-block:: bash

    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

- If you are using the CPU-only version of PyTorch, choose the +cpu installation link, such as:

.. code-block:: bash

    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cpu.html

**3. Troubleshooting Compatibility Issues**

If you encounter issues during installation, ensure the following:
- PyTorch is correctly installed and the version matches the selected torch-scatter wheel.
- Your Python and pip versions are up to date. You can update pip using:

.. code-block:: bash

    python -m pip install --upgrade pip

**4. Verify Installation**

After installation, verify that torch-scatter is installed successfully by running:

.. code-block:: bash

    python -c "import torch_scatter; print('torch-scatter installed successfully')"
