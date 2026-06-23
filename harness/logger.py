"""
Trajectory Logger — JSONL phase space recorder.

Records every step of the optimization trajectory as a JSONL file.
Each line is a complete TrajectoryStep serialized to JSON.

This enables post-hoc analysis of the full phase space trajectory:
- Loss curves
- Oscillation detection
- Phase transition identification
- Code evolution tracking
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.types import TrajectoryStep, EpisodeResult


class TrajectoryLogger:
    """
    Records trajectory steps to a JSONL file.

    Each line is a self-contained JSON object with:
    - step: int
    - state: {code, step, loss, metadata}
    - action: {patch, format, confidence, reasoning, metadata}
    - observation: {current_code, loss, passed_tests, ...}
    - reward: float
    - done: bool
    - dynamics_status: str
    - timestamp: str
    """

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear file on init
        with open(self.output_path, "w", encoding="utf-8") as f:
            pass

    def record(self, step: TrajectoryStep) -> None:
        """Append one step to the JSONL file."""
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(step.model_dump_json() + "\n")

    def record_episode_summary(self, result: EpisodeResult) -> None:
        """
        Append an episode summary line (marked with _type="summary").
        """
        summary = {
            "_type": "summary",
            "success": result.success,
            "total_steps": result.total_steps,
            "final_loss": result.final_loss,
            "min_loss": result.min_loss,
            "oscillation_index": result.oscillation_index,
            "wall_time_seconds": result.wall_time_seconds,
            "auditor_stats": result.auditor_stats,
            "agent_stats": result.agent_stats,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")


class ExperimentLogger:
    """
    Higher-level logger for an entire experiment (multiple episodes).

    Creates a directory structure:
        output_dir/
        ├── config.yaml              # experiment config
        ├── trajectories/
        │   ├── trial_0.jsonl        # per-step trajectory
        │   ├── trial_1.jsonl
        │   └── ...
        └── summary.json             # aggregated results
    """

    def __init__(self, output_dir: str | Path, experiment_name: str = "default"):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.trajectories_dir = self.output_dir / "trajectories"
        self.trajectories_dir.mkdir(parents=True, exist_ok=True)

    def get_trajectory_logger(self, trial: int) -> TrajectoryLogger:
        """Get a TrajectoryLogger for a specific trial."""
        path = self.trajectories_dir / f"trial_{trial}.jsonl"
        return TrajectoryLogger(path)

    def save_summary(self, results: List[EpisodeResult]) -> None:
        """Save aggregated experiment results."""
        summary = {
            "experiment_name": self.experiment_name,
            "num_trials": len(results),
            "success_rate": sum(1 for r in results if r.success) / len(results) if results else 0,
            "avg_steps": sum(r.total_steps for r in results) / len(results) if results else 0,
            "avg_final_loss": sum(r.final_loss for r in results) / len(results) if results else 0,
            "avg_oscillation_index": sum(r.oscillation_index for r in results) / len(results) if results else 0,
            "timestamp": datetime.now().isoformat(),
            "trials": [
                {
                    "trial": i,
                    "success": r.success,
                    "total_steps": r.total_steps,
                    "final_loss": r.final_loss,
                    "min_loss": r.min_loss,
                    "oscillation_index": r.oscillation_index,
                    "wall_time_seconds": r.wall_time_seconds,
                }
                for i, r in enumerate(results)
            ],
        }

        path = self.output_dir / "summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
