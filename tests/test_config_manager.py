"""
Tests for YAML config manager (from test_new_architecture.py).
"""

import os
from harness.config_manager import load_config, save_config, create_baseline_config, create_smo_config


def test_create_baseline_config():
    cfg = create_baseline_config(env_name="saddle", temperature=0.0, num_trials=3)
    assert cfg.agent.type == "baseline"
    assert cfg.agent.temperature == 0.0
    assert cfg.num_trials == 3
    assert cfg.environment.name == "saddle"


def test_create_smo_config():
    cfg = create_smo_config(env_name="convex", beta=0.8, obfuscation_level=2)
    assert cfg.agent.type == "momentum"
    assert cfg.agent.beta == 0.8
    assert cfg.environment.obfuscation_level == 2


def test_config_save_load_roundtrip(output_dir="outputs"):
    baseline_cfg = create_baseline_config(env_name="saddle", temperature=0.0, num_trials=3)
    test_path = f"{output_dir}/test_config.yaml"
    save_config(baseline_cfg, test_path)
    loaded = load_config(test_path)
    assert loaded.experiment_name == baseline_cfg.experiment_name
    assert loaded.agent.type == baseline_cfg.agent.type
    assert loaded.agent.temperature == baseline_cfg.agent.temperature


def test_load_existing_yaml():
    configs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
    yaml_path = os.path.join(configs_dir, "saddle_baseline.yaml")
    if os.path.exists(yaml_path):
        cfg = load_config(yaml_path)
        assert cfg.agent.type == "baseline"
        assert cfg.environment.name == "saddle"
