from .bridge_robot_env import BridgeRobotEnv, EnvConfig, RobotState, StepResult

__all__ = [
    "BridgeRobotEnv",
    "EnvConfig",
    "RobotState",
    "StepResult",
]

try:
    from .torque_control_env import TorqueControlEnv
except ImportError:  # pragma: no cover - allows core environment usage without RL extras
    TorqueControlEnv = None
else:
    __all__.append("TorqueControlEnv")

try:
    from .link_allocation_env import LinkAllocationConfig, LinkAllocationEnv
except ImportError:  # pragma: no cover - allows Phase 1 usage without RL extras
    LinkAllocationConfig = None
    LinkAllocationEnv = None
else:
    __all__.extend(["LinkAllocationConfig", "LinkAllocationEnv"])
