Environments
======================

算法库支持的仿真环境
----------------------
本软件包含的单智能体任务仿真环境有gym下的Atari，Mujoco，Classic Control，Box2D。
包含的多智能体任务仿真环境有PettingZoo开源环境下的MPE和SISL，StarCraft2，MAgent2，Google Football环境等。
每个仿真环境下包含丰富的任务场景，如下表所示。

用户自定义环境
----------------------

若用户所使用的仿真环境不在表1中，可对环境进行二次封装并存放至./environment目录下。具体添加方法如下：

步骤一：

在./environment目录下创建文件夹myenv（名称自拟），进入myenv文件件并创建my_env.py文件（文件名自拟），编写如下内容：

.. code-block:: python

    class My_Env(gym.Env):
        def __init__(env_id: str, seed: str)
            self.env = make_env(env_id)
            self.env.seed(seed)
            self.obeservation_space = Box(0, 1, self.env.dim_state)
        self.action_space = self.env.action_space
            self.metadata = self.env.metadata
            self.reward_range = self.env.reward_range
            self.spec = self.env.spec
            super(My_Env, self).__init__()

        def reset(self):
            return self.env.reset()

        def step(self, action):
            return self.env.step()
        def seed(self, seed):
            return self.env.seed(seed)
        def render(self, mode)
            return self.env.render(mode)
        def close(self)
            self.env.close()

步骤二：在./environment/__init__.py文件中导入自定义的环境类My_Env。
::

    from .myenv.my_env import My_Env

向量化仿真环境
----------------------
为了提高采样效率，节省算法运行时间，本软件支持向量化仿真环境设置，即运行多个仿真环境同时采样。
向量化环境基类VecEnv的定义位于./environment/vector_envs/vector_env.py文件中，
在此基类上定义继承类DummyVecEnv及DummyVecEnv_MAS，分别用于实现单智能体和多智能体向量化仿真环境，
代码位于./environment/vector_envs/dummy_vec_env.py文件中。
