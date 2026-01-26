How to Add a New Benchmark
======================================

本节介绍如何在 XuanCe 中添加一个新的基准测试。
在 XuanCe 中，一个基准测试由 **一种算法、一个环境场景以及多个随机种子** 共同定义。

Step 1: 选择基准测试任务
---------------------------------------

确定基准测试环境及场景任务。例如:

- Environment: Atari
- Scenario: Breakout-v5

如果对应的目录不存在，请创建该目录：

.. code-block::text

    benchmarks/Atari/Breakout-v5/

Step 2: 创建对应算法目录
---------------------------------------

在场景目录下，为该算法创建一个子目录:

.. code-block::text

     benchmarks/Atari/Breakout-v5/<algorithm>/


例如，针对 **PPO**算法:

.. code-block::text

    benchmarks/Atari/Breakout-v5/ppo/


Step 3: 准备算法配置（可选）
---------------------------------------

如果该算法需要特定的配置文件，请将其放置在算法目录中:

.. code-block::

    benchmarks/Atari/Breakout-v5/ppo/ppo_atari.yaml


该配置文件用于定义基准测试中使用的超参数以及与环境相关的设置。

Step 4: 编写基准测试脚本
---------------------------------------

创建基准测试脚本，文件命名如下:

.. code-block::text

    run_<algorithm>_<scenario>.sh

例如:

.. code-block::text

    run_iql_simple_spread_v3.sh

每个基准测试脚本应该:

- 调用共享的 train.py 脚本
- 运行多个随机种子（默认：5 个）
- 清晰标明每个随机种子的开始与结束
- 不重复输出 XuanCe 已经打印的算法或环境信息

结构示例:

.. code-block::bash

    #!/usr/bin/env bash
    set -euo pipefail

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
    PYTHON=python

    ALGO="ppo"
    ENV="Atari"
    ENV_ID="Breakout-v5"

    OUT_ROOT="${PROJECT_ROOT}/benchmarks/results/raw/${ENV}/${ENV_ID}/${ALGO}"

    for SEED in 1 2 3 4 5; do
      WORKDIR="${OUT_ROOT}/seed_${SEED}"
      mkdir -p "${WORKDIR}"

      echo "========== [Benchmark START] seed=${SEED} =========="
      START_TIME=$(date +%s)

      if ${PYTHON} "${PROJECT_ROOT}/train.py" \
          --algo "${ALGO}" \
          --env "${ENV}" \
          --env-id "${ENV_ID}" \
          --seed "${SEED}" \
          --workdir "${WORKDIR}"; then
        STATUS="SUCCESS"
      else
        STATUS="FAILED"
      fi

      END_TIME=$(date +%s)
      ELAPSED=$((END_TIME - START_TIME))

      echo "========== [Benchmark END] seed=${SEED} | status=${STATUS} | time=${ELAPSED}s =========="
    done

Step 5: 将该基准测试添加到套件脚本中（可选）
---------------------------------------

如果希望将新的基准测试纳入某个基准测试套件，请编辑场景目录下的套件脚本:

.. code-block::text

    benchmarks/Atai/Breakout-v5/run_simple_spread_all.sh


将新的基准测试脚本添加到列表中:

.. code-block::bash

    SCRIPTS=(
      "${ROOT_DIR}/dqn/run_dqn_Breakout-v5.sh"
      "${ROOT_DIR}/ppo/run_ppo_Breakout-v5.sh"
      "${ROOT_DIR}/<new_algo>/run_<new_algo>_Breakout-v5.sh"
    )

Step 6: 运行并验证
---------------------------------------

运行基准测试脚本:

.. code-block::bash

    bash benchmarks/Atari/Breakout-v5/iql/run_ppo_Breakout-v5.sh


验证如下信息:
- 所有随机种子按顺序依次运行
- 每个随机种子均清晰打印 START / END 标记
- 实验结果保存在正确的目录结构下
- 重新运行脚本可以复现实验结果

设计原则
---------------------------------------

在添加新的基准测试时，建议遵循以下原则：
- 一个脚本 = 一个基准测试
- 基准测试脚本是唯一的权威来源
- 避免硬编码绝对路径
- 避免重复输出 XuanCe 已经处理的日志信息
- 优先保证清晰性与可复现性
