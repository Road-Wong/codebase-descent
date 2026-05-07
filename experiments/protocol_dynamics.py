"""
Phase II: Dynamics Scan Protocol
Sweep hyperparameter grid and observe optimization dynamics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import ConvexEnvironment, SaddleEnvironment
from agent import LLMKernel, CodeOptimizer
from harness import NoiseInjector, DynamicsAuditor, ExperimentStatus
import json
import time
from typing import Dict, Any, List


def load_config():
    """Load API configuration."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.txt')
    with open(config_path, 'r') as f:
        lines = f.read().strip().split('\n')
        return {
            'base_url': lines[0],
            'model': lines[1],
            'api_key': lines[2]
        }


def run_trial(
    env,
    llm_kernel: LLMKernel,
    temperature: float,
    context_k: int,
    noise_level: int,
    max_steps: int = 20,
    seed: int = 0
) -> Dict[str, Any]:
    """
    Run a single trial with specific hyperparameters.
    
    Args:
        env: Environment
        llm_kernel: LLM kernel
        temperature: Sampling temperature
        context_k: Context window size
        noise_level: Difficulty level
        max_steps: Maximum optimization steps
        seed: Random seed
        
    Returns:
        Trajectory dictionary
    """
    # Initialize components
    injector = NoiseInjector(env.c_star)
    c_0, initial_distance = injector.perturb(noise_level, seed=seed)
    
    optimizer = CodeOptimizer(
        llm_kernel,
        context_window=context_k,
        temperature=temperature
    )
    
    auditor = DynamicsAuditor()
    
    # Task description
    task_description = "Implement the code to pass all tests. Fix any errors."
    
    # Trajectory storage
    trajectory = []
    code = c_0
    env.reset()
    
    # Optimization loop
    for t in range(max_steps):
        # Evaluate current code
        loss, info = env.step(code)
        
        # Audit dynamics
        status = auditor.step(loss, code)
        
        # Record step
        step_data = {
            't': t,
            'loss': loss,
            'passed_tests': info['passed_tests'],
            'total_tests': info['total_tests'],
            'status': status.value,
            'code_length': len(code)
        }
        trajectory.append(step_data)
        
        print(f"    Step {t}: Loss={loss:.3f}, Status={status.value}")
        
        # Check for early stopping
        if status == ExperimentStatus.SOLVED:
            print(f"    ✓ Solved in {t} steps!")
            break
        
        if status == ExperimentStatus.OSCILLATING and t > 10:
            print(f"    ⚠ Oscillation detected, stopping early")
            break
        
        if status == ExperimentStatus.STAGNATING and t > 10:
            print(f"    ⚠ Stagnation detected, stopping early")
            break
        
        # Generate next code
        try:
            code = optimizer.step(task_description, code, info)
        except Exception as e:
            print(f"    ✗ Error generating code: {e}")
            break
    
    # Get statistics
    stats = auditor.get_statistics()
    
    return {
        'temperature': temperature,
        'context_window': context_k,
        'noise_level': noise_level,
        'seed': seed,
        'initial_distance': initial_distance,
        'trajectory': trajectory,
        'statistics': stats,
        'converged': stats.get('converged', False),
        'total_steps': len(trajectory)
    }


def run_dynamics_scan(
    env_name: str = 'convex',
    noise_level: int = 2,
    trials_per_config: int = 3
):
    """
    Run full dynamics scan across hyperparameter grid.
    
    Args:
        env_name: 'convex' or 'saddle'
        noise_level: Difficulty level (from calibration)
        trials_per_config: Number of trials per configuration
    """
    print(f"=== Dynamics Scan: {env_name.upper()} Environment ===\n")
    print(f"Noise Level: {noise_level}")
    print(f"Trials per config: {trials_per_config}\n")
    
    # Load configuration
    config = load_config()
    
    # Initialize environment
    if env_name == 'convex':
        env = ConvexEnvironment()
    else:
        env = SaddleEnvironment()
    
    # Initialize LLM
    llm_kernel = LLMKernel(
        api_key=config['api_key'],
        base_url=config['base_url'],
        model=config['model']
    )
    
    # Hyperparameter grid
    temperatures = [0.1, 0.5, 0.8, 1.2]
    context_windows = [2, 5, 10]
    
    all_results = []
    
    # Sweep grid
    total_configs = len(temperatures) * len(context_windows)
    config_num = 0
    
    for temp in temperatures:
        for context_k in context_windows:
            config_num += 1
            print(f"\n[Config {config_num}/{total_configs}] T={temp}, K={context_k}")
            print("-" * 50)
            
            config_results = []
            
            for trial in range(trials_per_config):
                print(f"  Trial {trial + 1}/{trials_per_config}:")
                
                try:
                    result = run_trial(
                        env=env,
                        llm_kernel=llm_kernel,
                        temperature=temp,
                        context_k=context_k,
                        noise_level=noise_level,
                        max_steps=15,
                        seed=trial
                    )
                    config_results.append(result)
                    
                except Exception as e:
                    print(f"    ✗ Trial failed: {e}")
                    continue
                
                # Small delay to avoid rate limiting
                time.sleep(1)
            
            all_results.extend(config_results)
    
    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'trajectories',
        f'dynamics_scan_{env_name}_{int(time.time())}.json'
    )
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n=== Dynamics Scan Complete ===")
    print(f"Total trials: {len(all_results)}")
    print(f"Results saved to: {output_path}")
    
    # Summary statistics
    converged_count = sum(1 for r in all_results if r.get('converged', False))
    print(f"Convergence rate: {converged_count}/{len(all_results)} ({converged_count/len(all_results):.1%})")
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run dynamics scan protocol')
    parser.add_argument('--env', type=str, default='convex', choices=['convex', 'saddle'],
                       help='Environment to test')
    parser.add_argument('--noise', type=int, default=2,
                       help='Noise level (from calibration)')
    parser.add_argument('--trials', type=int, default=3,
                       help='Trials per configuration')
    
    args = parser.parse_args()
    run_dynamics_scan(args.env, args.noise, args.trials)
