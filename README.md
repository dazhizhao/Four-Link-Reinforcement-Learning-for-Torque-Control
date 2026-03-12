# 桥梁检测机器人控制优化

面向桥梁检测机器人控制优化的研究型工程项目。仓库当前包含两条并行能力：

- Phase 1 已完成的 `torque-based physics environment`，用于四自由度平面机械臂的 `reset / step / render` 最小闭环。
- 下一阶段主线 `link allocation RL environment`，用于在总杆长固定的约束下优化四连杆长度分配，使末端工作空间尽可能大。

## 当前主线目标

当前强化学习目标不再是直接学习关节力矩控制，而是：

> 在保持四根杆总长不变的前提下，控制各个杆件的长度分配，使四连杆机构末端的运动空间尽可能大。

这里不再使用“最大可达半径”作为奖励，因为在总长固定时，最大外半径恒等于总长，无法区分不同长度分配。当前采用的目标是：

- `outer_radius = sum(lengths) = total_length`
- `inner_radius = max(0, 2 * max(lengths) - total_length)`
- `workspace_area = pi * (outer_radius^2 - inner_radius^2)`
- `reward = workspace_area / (pi * total_length^2)`

该奖励在 `(0, 1]` 内，长度越均衡，可达环域越接近完整圆盘，奖励越高。

## 项目结构

```text
project/
├── README.md
├── requirements.yaml
├── configs/
│   ├── default.yaml
│   ├── link_allocation_env.yaml
│   └── train_rl.yaml
├── env/
│   ├── __init__.py
│   ├── bridge_robot_env.py
│   ├── dynamics.py
│   ├── kinematics.py
│   ├── link_allocation_env.py
│   └── reward.py
├── scripts/
│   ├── run_env.py
│   ├── train_rl.py
│   └── visualize_env.py
├── tests/
│   ├── test_dynamics.py
│   ├── test_env.py
│   ├── test_kinematics.py
│   ├── test_link_allocation_env.py
│   ├── test_run_env.py
│   ├── test_train_rl.py
│   └── test_visualization.py
└── visualization/
    ├── __init__.py
    ├── plots.py
    ├── render.py
    └── video.py
```

## Phase 1：已完成的 torque-based 环境闭环

这一部分已经完成并稳定，可继续作为物理环境基线使用：

- 内部环境类 `BridgeRobotEnv`
- 正运动学 `env/kinematics.py`
- 简化动力学 `env/dynamics.py`
- 奖励函数 `env/reward.py`
- 单帧渲染与时序图 `visualization/`
- 单次 rollout 数据归档 `run_env_rollout.npz`
- 单次 rollout 回放视频 `rollout.mp4`
- smoke rollout 脚本 `scripts/run_env.py`
- 可视化脚本 `scripts/visualize_env.py`

### Phase 1 默认配置

`configs/default.yaml` 当前固化：

- `dt = 0.02`
- `max_steps = 250`
- `gravity = 9.81`
- `link_lengths = [1.2, 1.0, 0.8, 0.6]`
- `link_masses = [8.0, 6.0, 4.0, 2.0]`
- `payload_mass = 1.5`
- `joint_damping = [1.2, 1.0, 0.8, 0.6]`
- `torque_limits = [40.0, 30.0, 20.0, 10.0]`
- `home_pose = [0.2, 0.4, 0.2, 0.0]`
- `success_tolerance = 0.08`
- `output_dir = /openbayes/home/results`

### Phase 1 环境接口

```python
from env.bridge_robot_env import BridgeRobotEnv

env = BridgeRobotEnv()
obs = env.reset(seed=7)
step_result = env.step([0.0, 0.0, 0.0, 0.0])
fig = env.render()
env.close()
```

### Phase 1 运行命令

运行环境 smoke rollout：

```bash
python scripts/run_env.py --policy zero --seed 7 --output-dir /openbayes/home/results
python scripts/run_env.py --policy random --seed 7 --output-dir /openbayes/home/results
```

生成姿态图、时序图和视频：

```bash
python scripts/visualize_env.py
```

执行测试：

```bash
python -m pytest tests
```

Phase 1 相关输出默认写入：

```text
/openbayes/home/results
```

不要把运行结果写回代码目录。

## Phase 2：杆长分配 RL 环境

当前 RL 主线使用独立环境 `LinkAllocationEnv`，不复用 `BridgeRobotEnv` 的动力学 rollout。

### 任务定义

- 状态：固定 13 维观测，包含默认杆长、上下界和总长信息。
- 动作：4 维候选杆长。
- 约束：动作会被投影到“逐杆上下界 + 总长固定”的 bounded simplex 上。
- 回合：单步 episode，`step()` 后立即结束。
- 奖励：归一化工作空间面积。

### 默认环境配置

`configs/link_allocation_env.yaml` 当前为：

- `total_length = 3.6`
- `default_link_lengths = [1.2, 1.0, 0.8, 0.6]`
- `min_link_lengths = [0.4, 0.4, 0.4, 0.4]`
- `max_link_lengths = [1.4, 1.4, 1.4, 1.4]`

### `LinkAllocationEnv` 接口约定

```python
from env.link_allocation_env import LinkAllocationEnv

env = LinkAllocationEnv()
obs, info = env.reset(seed=7)
obs, reward, terminated, truncated, info = env.step([0.9, 0.9, 0.9, 0.9])
```

`info` 至少包含：

- `allocated_lengths`
- `raw_action`
- `workspace_area`
- `inner_radius`
- `outer_radius`
- `projection_applied`

## SAC 训练入口

`scripts/train_rl.py` 已从占位脚本扩展为杆长分配任务的 SAC 训练入口。

### 默认训练配置

`configs/train_rl.yaml` 当前包含：

- `algo = sac`
- `policy = MlpPolicy`
- `total_timesteps = 2000`
- `seed = 7`
- `device = auto`
- `learning_starts = 32`
- `buffer_size = 4096`
- `batch_size = 64`
- `train_freq = 1`
- `gradient_steps = 1`
- `learning_rate = 3e-4`
- `gamma = 0.99`
- `tau = 0.005`
- `eval_episodes = 5`
- `run_name = sac_link_alloc`
- `output_dir = /openbayes/home/results`

### 训练命令

使用默认配置：

```bash
python scripts/train_rl.py
```

覆盖总步数、输出目录和运行名：

```bash
python scripts/train_rl.py \
  --total-timesteps 5000 \
  --run-name exp001 \
  --output-dir /openbayes/home/results
```

### 训练输出

训练结果写入：

```text
<output_dir>/rl_link_alloc/<run_name>/
```

至少包含：

- `model_final.zip`
- `train_config.json`
- `evaluation.json`
- `best_lengths.json`
- `monitor.csv`

其中：

- `evaluation.json` 记录 deterministic evaluation 的均值奖励、最优奖励和面积指标。
- `best_lengths.json` 记录当前训练得到的最佳杆长分配。

## 开发方式

项目仍采用“本地开发，云端 Linux 执行”的工作方式：

- 本地负责编辑代码、整理文档、维护配置和提交版本。
- 云端负责安装依赖、运行脚本、保存结果和训练。
- 默认结果目录仍为 `/openbayes/home/results`。

推荐的云端持久化目录：

```text
/openbayes/home/
├── Project_jiaogai/
├── results/
└── envs/
```

创建并激活环境：

```bash
mkdir -p /openbayes/home/Project_jiaogai
mkdir -p /openbayes/home/results
mkdir -p /openbayes/home/envs

conda env create -p /openbayes/home/envs/bridge-robot-cloud -f /openbayes/home/Project_jiaogai/requirements.yaml
conda activate /openbayes/home/envs/bridge-robot-cloud
```

如果镜像中 `conda activate` 不可直接使用：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /openbayes/home/envs/bridge-robot-cloud
```

## 当前状态

- 已完成：Phase 1 torque-based 最小环境骨架、简化动力学、基础渲染、脚本入口和测试骨架。
- 已新增：杆长分配 RL 环境 `LinkAllocationEnv`。
- 已新增：SAC 训练入口 `scripts/train_rl.py`。
- 已新增：杆长分配环境与训练 smoke test。
- 当前主线：在固定总长约束下优化四杆长度分配，提升末端工作空间面积。
