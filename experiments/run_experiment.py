"""
Unified experiment runner using the new RL architecture.

Usage:
    # Run with YAML config
    python experiments/run_experiment.py --config configs/saddle_baseline.yaml

    # Run with CLI overrides
    python experiments/run_experiment.py --env saddle --agent baseline --trials 3

    # Run stress test (Baseline vs SMO at multiple obfuscation levels)
    python experiments/run_experiment.py --stress --env saddle --trials 5
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime
from pathlib import Path

from harness.config_manager import (
    load_config,
    create_baseline_config,
    create_smo_config,
)
from harness.loop_runner import (
    LoopRunner,
    create_env_from_config,
    create_agent_from_config,
)
from harness.auditor import DynamicsAuditor
from harness.logger import ExperimentLogger
from core.types import ExperimentConfig


def load_llm_kernel(config_path: str = "config.txt"):
    """Load LLM kernel from config.txt."""
    from agents import LLMKernel

    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path)
    with open(config_path, "r") as f:
        lines = f.read().strip().split("\n")
    return LLMKernel(
        api_key=lines[2],
        base_url=lines[0],
        model=lines[1],
    )


def run_single_config(config: ExperimentConfig, llm_kernel=None) -> list:
    """Run a single experiment configuration."""
    env = create_env_from_config(config)
    agent = create_agent_from_config(config, llm_kernel)
    auditor = DynamicsAuditor()
    logger = ExperimentLogger(config.output_dir, config.experiment_name)

    results = []
    for trial in range(config.num_trials):
        print(f"\n  Trial {trial + 1}/{config.num_trials}...")

        traj_logger = logger.get_trajectory_logger(trial)
        runner = LoopRunner(
            env=env,
            agent=agent,
            auditor=auditor,
            logger=traj_logger,
            max_steps=config.environment.max_steps,
        )
        result = runner.run()
        results.append(result)

        status = "SOLVED" if result.success else "FAILED"
        print(f"    {status}: loss={result.final_loss:.3f}, steps={result.total_steps}, OI={result.oscillation_index:.2f}")

    logger.save_summary(results)
    return results


def run_stress_test(env_name: str, num_trials: int, llm_kernel=None):
    """
    Run stress test: Baseline vs SMO at multiple obfuscation levels.
    This is Experiment 1 from the paper.
    """
    print("=" * 60)
    print("STRESS TEST: Baseline vs SMO")
    print("=" * 60)

    all_results = {}

    for level in [0, 1, 2]:
        print(f"\n--- Obfuscation Level {level} ---")

        for agent_type in ["baseline", "momentum"]:
            print(f"\n  [{agent_type}]")
            if agent_type == "baseline":
                config = create_baseline_config(
                    env_name=env_name,
                    obfuscation_level=level,
                    num_trials=num_trials,
                    max_steps=15,
                )
            else:
                config = create_smo_config(
                    env_name=env_name,
                    obfuscation_level=level,
                    num_trials=num_trials,
                    max_steps=15,
                )

            config.output_dir = f"outputs/stress_test/level{level}_{agent_type}"
            results = run_single_config(config, llm_kernel)
            all_results[(level, agent_type)] = results

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for level in [0, 1, 2]:
        print(f"\nLevel {level}:")
        for agent_type in ["baseline", "momentum"]:
            results = all_results.get((level, agent_type), [])
            if results:
                success_rate = sum(1 for r in results if r.success) / len(results) * 100
                avg_oi = sum(r.oscillation_index for r in results) / len(results)
                print(f"  {agent_type:10s}: Success={success_rate:5.1f}%, OI={avg_oi:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run Codebase Descent experiments")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--env", type=str, default="saddle", help="Environment name")
    parser.add_argument("--agent", type=str, default="baseline", choices=["baseline", "momentum"])
    parser.add_argument("--trials", type=int, default=1, help="Number of trials")
    parser.add_argument("--steps", type=int, default=20, help="Max steps per episode")
    parser.add_argument("--obfuscation", type=int, default=0, help="Obfuscation level (0-3)")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--beta", type=float, default=0.7, help="Momentum coefficient (SMO only)")
    parser.add_argument("--stress", action="store_true", help="Run stress test")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()

    # Load LLM kernel (needed for agent creation)
    try:
        llm_kernel = load_llm_kernel()
        print("LLM kernel loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load LLM kernel: {e}")
        print("Running in dry-run mode (no LLM calls).")
        llm_kernel = None

    if args.stress:
        run_stress_test(args.env, args.trials, llm_kernel)
        return

    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        temp = args.temperature or (0.0 if args.agent == "baseline" else 0.7)
        if args.agent == "baseline":
            config = create_baseline_config(
                env_name=args.env,
                temperature=temp,
                obfuscation_level=args.obfuscation,
                num_trials=args.trials,
                max_steps=args.steps,
            )
        else:
            config = create_smo_config(
                env_name=args.env,
                temperature=temp,
                beta=args.beta,
                obfuscation_level=args.obfuscation,
                num_trials=args.trials,
                max_steps=args.steps,
            )
        config.output_dir = args.output

    print(f"\nExperiment: {config.experiment_name}")
    print(f"Agent: {config.agent.type}, Env: {config.environment.name}")
    print(f"Trials: {config.num_trials}, Max steps: {config.environment.max_steps}")
    print(f"Obfuscation: {config.environment.obfuscation_level}")

    results = run_single_config(config, llm_kernel)

    # Print summary
    success_rate = sum(1 for r in results if r.success) / len(results) * 100 if results else 0
    avg_steps = sum(r.total_steps for r in results) / len(results) if results else 0
    print(f"\nDone: {success_rate:.0f}% success, avg {avg_steps:.1f} steps")


if __name__ == "__main__":
    main()
