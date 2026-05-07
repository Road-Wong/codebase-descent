"""
Phase I: Calibration Protocol
Find the difficulty sweet spot where greedy fails but annealing succeeds.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import ConvexEnvironment, SaddleEnvironment
from agent import LLMKernel, CodeOptimizer
from harness import NoiseInjector
import json


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


def test_difficulty_level(env, llm_kernel, difficulty_level: int, trials: int = 5):
    """
    Test a specific difficulty level.
    
    Args:
        env: Environment to test
        llm_kernel: LLM kernel
        difficulty_level: 1, 2, or 3
        trials: Number of trials
        
    Returns:
        Dictionary with success rates
    """
    injector = NoiseInjector(env.c_star)
    
    greedy_successes = 0
    annealed_successes = 0
    
    task_description = "Implement the code to pass all tests. Fix any errors."
    
    for trial in range(trials):
        print(f"  Trial {trial + 1}/{trials}...")
        
        # Test with greedy (T=0.1)
        perturbed_code, distance = injector.perturb(difficulty_level, seed=trial)
        optimizer = CodeOptimizer(llm_kernel, context_window=2, temperature=0.1)
        
        code = perturbed_code
        env.reset()
        
        for step in range(5):
            loss, info = env.step(code)
            if env.is_solved(code):
                greedy_successes += 1
                break
            code = optimizer.step(task_description, code, info)
        
        # Test with annealing (T=0.7)
        perturbed_code, distance = injector.perturb(difficulty_level, seed=trial + 100)
        optimizer = CodeOptimizer(llm_kernel, context_window=5, temperature=0.7)
        
        code = perturbed_code
        env.reset()
        
        for step in range(10):
            loss, info = env.step(code)
            if env.is_solved(code):
                annealed_successes += 1
                break
            code = optimizer.step(task_description, code, info)
    
    greedy_rate = greedy_successes / trials
    annealed_rate = annealed_successes / trials
    
    return {
        'difficulty_level': difficulty_level,
        'greedy_success_rate': greedy_rate,
        'annealed_success_rate': annealed_rate,
        'is_sweet_spot': greedy_rate < 0.2 and annealed_rate > 0.8
    }


def run_calibration(env_name: str = 'convex'):
    """
    Run calibration protocol to find optimal difficulty level.
    
    Args:
        env_name: 'convex' or 'saddle'
    """
    print(f"=== Calibration Protocol: {env_name.upper()} Environment ===\n")
    
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
    
    results = []
    
    # Test each difficulty level
    for level in [1, 2, 3]:
        print(f"\nTesting Difficulty Level {level}...")
        result = test_difficulty_level(env, llm_kernel, level, trials=3)
        results.append(result)
        
        print(f"  Greedy (T=0.1) success rate: {result['greedy_success_rate']:.2%}")
        print(f"  Annealed (T=0.7) success rate: {result['annealed_success_rate']:.2%}")
        print(f"  Sweet spot: {'YES' if result['is_sweet_spot'] else 'NO'}")
    
    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        f'calibration_{env_name}.json'
    )
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Calibration Complete ===")
    print(f"Results saved to: {output_path}")
    
    # Recommend difficulty level
    sweet_spots = [r for r in results if r['is_sweet_spot']]
    if sweet_spots:
        recommended = sweet_spots[0]['difficulty_level']
        print(f"Recommended difficulty level: {recommended}")
    else:
        print("Warning: No sweet spot found. May need to adjust parameters.")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run calibration protocol')
    parser.add_argument('--env', type=str, default='convex', choices=['convex', 'saddle'],
                       help='Environment to calibrate')
    
    args = parser.parse_args()
    run_calibration(args.env)
