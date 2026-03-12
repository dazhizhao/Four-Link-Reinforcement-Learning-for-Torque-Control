from __future__ import annotations

from datetime import datetime
from pathlib import Path
import secrets


def build_run_dir(output_root: Path, task_dir: str, run_name: str) -> tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(2)
    resolved_run_name = f"{run_name}_{timestamp}_{suffix}"
    run_dir = output_root / task_dir / resolved_run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, resolved_run_name


def ensure_artifacts_dir(run_dir: Path) -> Path:
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir
