"""
Stress Test Experiment — v2 (Architecture-Compliant)

Uses the full LoopRunner + SaddleEnvironment + DiffAgent pipeline.
SGD single-sample feedback, SEARCH/REPLACE protocol, proper obfuscation.

Usage:
    python experiments/run_stress_test.py --noise 1 --trials 5
    python experiments/run_stress_test.py --noise 2 --trials 3
    python experiments/run_stress_test.py --noise 1 2 --trials 5
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.llm_kernel import LLMKernel
from agents.momentum_agent import SemanticMomentumOptimizer, BaselineAgent
from environments.env_saddle import SaddleEnvironment
from harness.loop_runner import LoopRunner
from harness.auditor import DynamicsAuditor


def load_config():
    """Load API configuration."""
    with open('config.txt', 'r') as f:
        lines = f.read().strip().split('\n')
        return {
            'base_url': lines[0],
            'model': lines[1],
            'api_key': lines[2]
        }


def run_single_trial(
    agent_type: str,
    env: SaddleEnvironment,
    llm_kernel: LLMKernel,
    temperature: float,
    beta: float,
    max_steps: int = 20,
) -> Dict[str, Any]:
    """
    Run a single episode using the full LoopRunner pipeline.

    Returns:
        Episode result with trajectory, agent_stats (including patch_success_rate).
    """
    # Create agent
    if agent_type == 'baseline':
        agent = BaselineAgent(llm_kernel, temperature=temperature)
    else:
        agent = SemanticMomentumOptimizer(
            llm_kernel, beta=beta, temperature=temperature
        )

    # Create auditor and runner
    auditor = DynamicsAuditor()
    runner = LoopRunner(
        env=env,
        agent=agent,
        auditor=auditor,
        max_steps=max_steps,
    )

    # Run episode
    result = runner.run()

    # Extract key metrics
    agent_stats = result.agent_stats or {}
    auditor_stats = result.auditor_stats or {}

    # Build per-step loss trajectory
    loss_trajectory = [ts.observation.loss for ts in result.trajectory]

    return {
        'agent_type': agent_type,
        'success': result.success,
        'total_steps': result.total_steps,
        'final_loss': result.final_loss,
        'min_loss': result.min_loss,
        'oscillation_index': result.oscillation_index,
        'patch_success_rate': agent_stats.get('patch_success_rate', 0.0),
        'valid_patches': agent_stats.get('valid_patches', 0),
        'invalid_patches': agent_stats.get('invalid_patches', 0),
        'loss_trajectory': loss_trajectory,
        'wall_time': result.wall_time_seconds,
    }


def run_experiment(
    noise_levels: List[int],
    num_trials: int,
    temperature: float,
    beta: float,
    max_steps: int,
):
    """
    Run the full diagnostic experiment.

    For each noise level, runs Baseline and SMO agents for num_trials each.
    """
    config = load_config()
    llm_kernel = LLMKernel(
        api_key=config['api_key'],
        base_url=config['base_url'],
        model=config['model'],
        timeout=60.0,
        max_retries=2,
    )

    print("=" * 70)
    print("STRESS TEST v2 — Architecture-Compliant Diagnostic")
    print("=" * 70)
    print(f"  Environment:  Saddle (26 tests, lexer+parser interpreter)")
    print(f"  Noise levels: {noise_levels}")
    print(f"  Trials:       {num_trials}")
    print(f"  Temperature:  {temperature}")
    print(f"  Beta (SMO):   {beta}")
    print(f"  Max steps:    {max_steps}")
    print(f"  Model:        {config['model']}")
    print()

    all_results = []

    for noise in noise_levels:
        print(f"\n{'='*70}")
        print(f"NOISE LEVEL {noise}")
        print('='*70)

        for agent_type in ['baseline', 'momentum']:
            trial_results = []

            for trial in range(num_trials):
                print(f"\n--- {agent_type.upper()} | Noise {noise} | Trial {trial+1}/{num_trials} ---")

                # Create a fresh environment for each trial
                env = SaddleEnvironment(obfuscation_level=noise, max_steps=max_steps)

                result = run_single_trial(
                    agent_type=agent_type,
                    env=env,
                    llm_kernel=llm_kernel,
                    temperature=temperature,
                    beta=beta,
                    max_steps=max_steps,
                )
                result['noise_level'] = noise
                result['trial'] = trial

                trial_results.append(result)
                all_results.append(result)

                # Print per-trial summary
                print(f"  Final Loss: {result['final_loss']:.3f}")
                print(f"  Min Loss:   {result['min_loss']:.3f}")
                print(f"  Steps:      {result['total_steps']}")
                print(f"  Patch Success Rate: {result['patch_success_rate']:.1%}")
                print(f"  OI:         {result['oscillation_index']:.3f}")
                print(f"  Time:       {result['wall_time']:.1f}s")

            # Print aggregate for this agent+noise combo
            _print_aggregate(agent_type, noise, trial_results)

    # Save results
    _save_results(all_results)

    return all_results


def _print_aggregate(agent_type: str, noise: int, trials: List[Dict]):
    """Print aggregate statistics for one agent+noise configuration."""
    n = len(trials)
    if n == 0:
        return

    avg_final_loss = sum(t['final_loss'] for t in trials) / n
    avg_min_loss = sum(t['min_loss'] for t in trials) / n
    avg_psr = sum(t['patch_success_rate'] for t in trials) / n
    avg_oi = sum(t['oscillation_index'] for t in trials) / n
    success_count = sum(1 for t in trials if t['success'])

    # Average loss trajectory (pad to same length)
    max_len = max(len(t['loss_trajectory']) for t in trials)
    avg_trajectory = []
    for i in range(max_len):
        vals = [t['loss_trajectory'][i] for t in trials if i < len(t['loss_trajectory'])]
        avg_trajectory.append(sum(vals) / len(vals) if vals else 0.0)

    print(f"\n  +-- AGGREGATE: {agent_type.upper()} @ Noise {noise} ({n} trials) --+")
    print(f"  | Success Rate:         {success_count}/{n} ({success_count/n*100:.0f}%)")
    print(f"  | Avg Final Loss:       {avg_final_loss:.3f}")
    print(f"  | Avg Min Loss:         {avg_min_loss:.3f}")
    print(f"  | Avg Patch Success:    {avg_psr:.1%}")
    print(f"  | Avg Oscillation Idx:  {avg_oi:.3f}")
    print(f"  | Avg Loss Trajectory:  {['%.2f' % v for v in avg_trajectory]}")
    print(f"  +{'-'*50}+")


def _save_results(results: List[Dict]):
    """Save results to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"stress_test_v2_{timestamp}.json")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Stress Test v2 — Diagnostic Experiment")
    parser.add_argument('--noise', type=int, nargs='+', default=[1, 2],
                        help='Obfuscation levels (0=clean, 1=rename, 2=visual, 3=extreme)')
    parser.add_argument('--trials', type=int, default=3,
                        help='Number of trials per agent+noise combo')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='LLM sampling temperature')
    parser.add_argument('--beta', type=float, default=0.6,
                        help='SMO momentum coefficient (β)')
    parser.add_argument('--max-steps', type=int, default=20,
                        help='Max steps per episode')

    args = parser.parse_args()

    print(f"\nRunning diagnostic experiment...")
    print(f"  Noise levels: {args.noise}")
    print(f"  Trials:       {args.trials}")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Beta (SMO):   {args.beta}")
    print()

    try:
        results = run_experiment(
            noise_levels=args.noise,
            num_trials=args.trials,
            temperature=args.temperature,
            beta=args.beta,
            max_steps=args.max_steps,
        )
        print("\n[OK] Experiment completed successfully!")
    except Exception as e:
        print(f"\n[FAIL] Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
