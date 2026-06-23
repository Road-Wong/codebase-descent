"""
YAML-driven configuration manager.

Loads experiment configs from YAML files and validates them
via Pydantic's ExperimentConfig model.

Usage:
    config = load_config("configs/saddle_baseline.yaml")
    env = create_env_from_config(config)
    agent = create_agent_from_config(config, llm_kernel)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from core.types import (
    AgentConfig,
    EnvironmentConfig,
    ExperimentConfig,
)


def load_config(path: str | Path) -> ExperimentConfig:
    """
    Load an ExperimentConfig from a YAML file.

    Args:
        path: Path to YAML config file.

    Returns:
        Validated ExperimentConfig instance.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the YAML is invalid or doesn't match the schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(raw)}")

    return _parse_config(raw)


def _parse_config(raw: Dict[str, Any]) -> ExperimentConfig:
    """Parse a raw dict into ExperimentConfig, filling defaults."""
    agent_raw = raw.get("agent", {})
    env_raw = raw.get("environment", {})

    agent_cfg = AgentConfig(**agent_raw) if agent_raw else AgentConfig()
    env_cfg = EnvironmentConfig(**env_raw) if env_raw else EnvironmentConfig()

    return ExperimentConfig(
        experiment_name=raw.get("experiment_name", "default"),
        num_trials=raw.get("num_trials", 1),
        agent=agent_cfg,
        environment=env_cfg,
        seed=raw.get("seed"),
        output_dir=raw.get("output_dir", "outputs"),
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """
    Save an ExperimentConfig to a YAML file.

    Args:
        config: The config to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump()
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Pre-built configs for common experiments
# ---------------------------------------------------------------------------

def create_baseline_config(
    env_name: str = "saddle",
    temperature: float = 0.0,
    obfuscation_level: int = 0,
    num_trials: int = 1,
    max_steps: int = 20,
) -> ExperimentConfig:
    """Create a baseline agent config."""
    return ExperimentConfig(
        experiment_name=f"baseline_{env_name}",
        num_trials=num_trials,
        agent=AgentConfig(type="baseline", temperature=temperature),
        environment=EnvironmentConfig(
            name=env_name,
            obfuscation_level=obfuscation_level,
            max_steps=max_steps,
        ),
    )


def create_smo_config(
    env_name: str = "saddle",
    temperature: float = 0.7,
    beta: float = 0.7,
    obfuscation_level: int = 0,
    num_trials: int = 1,
    max_steps: int = 20,
) -> ExperimentConfig:
    """Create an SMO (momentum) agent config."""
    return ExperimentConfig(
        experiment_name=f"smo_{env_name}",
        num_trials=num_trials,
        agent=AgentConfig(type="momentum", temperature=temperature, beta=beta),
        environment=EnvironmentConfig(
            name=env_name,
            obfuscation_level=obfuscation_level,
            max_steps=max_steps,
        ),
    )
