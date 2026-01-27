基准测试
======================================

XuanCe 提供了一套标准化的 benchmark 流程，用于在可复现的实验设置下
评估强化学习算法的性能。

为保持核心代码仓库的轻量化，**官方 benchmark 结果被单独发布并维护**
在以下仓库中：

- **XuanCe Benchmarks**：`https://github.com/agi-brain/xuance-benchmarks.git <https://github.com/agi-brain/xuance-benchmarks.git>`_

该仓库包含：
- 多种环境与算法的 benchmark 评测结果；
- 学习曲线与汇总图表；
- 用于复现实验的配置文件与元数据；
- 预训练模型（最佳 checkpoint）。

如果用户仅关注 benchmark 结果本身，可直接查阅上述仓库，而无需在本地重新运行实验。

目录结构
---------------------------------------

基准测试按照 **环境 → 场景 → 算法** 的层级结构进行组织:

.. code-block:: text

    benchmarks/
    ├── MPE/
    │ └── simple_spread_v3/
    │ ├── iql/
    │ │ ├── iql_simple_spread_v3.yaml
    │ │ └── run_iql_simple_spread_v3.sh
    │ ├── qmix/
    │ │ ├── qmix.yaml
    │ │ └── run_qmix_simple_spread_v3.sh
    │ ├── vdn/
    │ │ ├── vdn.yaml
    │ │ └── run_vdn_simple_spread_v3.sh
    │ └── run_simple_spread_all.sh


- 每个算法专用脚本（run_*.sh）定义一个基准测试。
- 套件脚本（例如 run_simple_spread_all.sh）用于在同一任务上依次运行多个算法。

运行单个基准测试
---------------------------------------

每个基准测试脚本都会使用多个随机种子（默认：5 个）来运行同一任务。

示例：在MPE环境的simple_spread_v3场景下，运行MADDPG基准测试脚本

.. code-block:: bash

    bash benchmarks/MPE/simple_spread_v3/iql/run_iql_simple_spread_v3.sh


在执行过程中，XuanCe 会输出算法、环境以及评测相关的信息，而基准测试脚本则负责为每个随机种子打印清晰的 START / END 边界标识。

运行基准测试套件
---------------------------------------

要在指定任务上评估所有已支持的算法，请使用对应的套件脚本：

.. code-block:: bash

    bash benchmarks/MPE/simple_spread_v3/run_simple_spread_all.sh

该脚本会在相同的环境和一致的评测设置下，按顺序依次运行 Algorithm_1、Algorithm_2、...、Algorithm_N。

评测规则
---------------------------------------

所有基准测试均遵循统一的评测规则：
- 使用不同随机种子进行多次独立运行
- 在训练过程中进行周期性评测（约每完成总步数的 1% 评测一次）
- 每次评测包含多个测试回合
- 报告的性能指标为回合回报的平均值
- 最终的基准测试结果在不同随机种子之间进行汇总

该设计确保了算法比较的公平性以及性能评估的稳健性。

基准测试结果
---------------------------------------

基准测试结果以结构化的目录布局进行存储：

.. code-block:: text

    (To be stored)

- 每个 learning_curve.csv 文件包含一个随机种子的学习曲线
- 可使用分析脚本生成汇总结果（不同随机种子上的均值 ± 标准差）

    TensorBoard 日志主要用于可视化和调试，而 CSV 文件被视为 **官方的基准测试结果产物**。

可复现性
---------------------------------------

为确保结果的可复现性，基准测试脚本会明确指定以下内容：

- 算法名称
- 环境与场景 ID
- 随机种子
- 训练与评测设置

基准测试脚本是所有已报告结果的唯一权威来源。
