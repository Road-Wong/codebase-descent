"""
Stress Test Experiment - The Core of Our Paper

This script implements Experiment 1 from the paper:
"Prove that Baseline agents collapse under de-semanticization, while SMO remains robust."

Protocol:
1. Take a simple coding task
2. Obfuscate it at different levels (0, 1, 2)
3. Run both Baseline and SMO agents
4. Compare success rate and oscillation index
"""

import json
import os
import sys
from typing import Dict, Any, List
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import obfuscate_code
from agent import LLMKernel, SemanticMomentumOptimizer, BaselineAgent
from harness import DynamicsAuditor


def load_config():
    """Load API configuration."""
    with open('config.txt', 'r') as f:
        lines = f.read().strip().split('\n')
        return {
            'base_url': lines[0],
            'model': lines[1],
            'api_key': lines[2]
        }


def create_simple_task():
    """
    Create a simple coding task for testing.
    
    This is intentionally simple so we can see the effect of obfuscation clearly.
    """
    task_description = """Implement a function that calculates the sum of all even numbers in a list.

Function signature: def sum_even_numbers(numbers: list) -> int

Examples:
- sum_even_numbers([1, 2, 3, 4, 5, 6]) should return 12 (2+4+6)
- sum_even_numbers([1, 3, 5]) should return 0
- sum_even_numbers([]) should return 0
- sum_even_numbers([2, 4, 6, 8]) should return 20
"""
    
    # Ground truth implementation
    clean_code = """def sum_even_numbers(numbers):
    total = 0
    for num in numbers:
        if num % 2 == 0:
            total += num
    return total
"""
    
    # Test cases
    test_cases = [
        ([1, 2, 3, 4, 5, 6], 12),
        ([1, 3, 5], 0),
        ([], 0),
        ([2, 4, 6, 8], 20),
        ([10, 15, 20, 25], 30),
        ([-2, -4, 3, 5], -6),
    ]
    
    return task_description, clean_code, test_cases


def evaluate_code(code: str, test_cases: List[tuple]) -> Dict[str, Any]:
    """
    Evaluate code against test cases.
    
    Returns:
        Dictionary with evaluation results
    """
    passed = 0
    total = len(test_cases)
    errors = []
    
    try:
        # Execute the code
        namespace = {}
        exec(code, namespace)
        
        # Check if function exists
        if 'sum_even_numbers' not in namespace:
            return {
                'passed_tests': 0,
                'total_tests': total,
                'errors': ['Function sum_even_numbers not found'],
                'has_syntax_error': True
            }
        
        func = namespace['sum_even_numbers']
        
        # Run test cases
        for i, (input_data, expected) in enumerate(test_cases):
            try:
                result = func(input_data)
                if result == expected:
                    passed += 1
                else:
                    errors.append(f"Test {i+1}: Expected {expected}, got {result}")
            except Exception as e:
                errors.append(f"Test {i+1}: {str(e)}")
                
    except SyntaxError as e:
        errors.append(f"Syntax error: {str(e)}")
        return {
            'passed_tests': 0,
            'total_tests': total,
            'errors': errors,
            'has_syntax_error': True
        }
    except Exception as e:
        errors.append(f"Execution error: {str(e)}")
        return {
            'passed_tests': 0,
            'total_tests': total,
            'errors': errors,
            'has_syntax_error': True
        }
    
    return {
        'passed_tests': passed,
        'total_tests': total,
        'errors': errors[:3],  # Limit to first 3 errors
        'has_syntax_error': False
    }


def run_single_trial(
    agent_type: str,
    task_description: str,
    initial_code: str,
    test_cases: List[tuple],
    obfuscation_level: int,
    llm_kernel: LLMKernel,
    max_steps: int = 15
) -> Dict[str, Any]:
    """
    Run a single trial with given agent and obfuscation level.
    
    Args:
        agent_type: 'baseline' or 'momentum'
        task_description: Task description
        initial_code: Starting code (possibly obfuscated)
        test_cases: Test cases for evaluation
        obfuscation_level: 0, 1, or 2
        llm_kernel: LLM interface
        max_steps: Maximum optimization steps
        
    Returns:
        Trial results dictionary
    """
    print(f"\n{'='*70}")
    print(f"Running {agent_type.upper()} agent (Level {obfuscation_level})")
    print('='*70)
    
    # Initialize agent
    if agent_type == 'baseline':
        agent = BaselineAgent(llm_kernel, temperature=0.0)
    else:  # momentum
        agent = SemanticMomentumOptimizer(llm_kernel, beta=0.7, temperature=0.7)
    
    # Initialize auditor
    auditor = DynamicsAuditor()
    
    # Optimization loop
    code = initial_code
    trajectory = []
    
    for step in range(max_steps):
        # Evaluate current code
        feedback = evaluate_code(code, test_cases)
        loss = 1.0 - (feedback['passed_tests'] / feedback['total_tests'])
        
        # Audit dynamics
        status = auditor.step(loss, code)
        
        # Record step
        step_data = {
            'step': step,
            'loss': loss,
            'passed_tests': feedback['passed_tests'],
            'total_tests': feedback['total_tests'],
            'status': status.value,
            'code_length': len(code)
        }
        trajectory.append(step_data)
        
        print(f"  Step {step}: Loss={loss:.3f}, Tests={feedback['passed_tests']}/{feedback['total_tests']}, Status={status.value}")
        
        # Check if solved
        if loss == 0.0:
            print(f"  ✓ Solved in {step} steps!")
            break
        
        # Check for early stopping (stagnation or severe oscillation)
        if step > 5:
            if status == DynamicsAuditor.ExperimentStatus.STAGNATING:
                print(f"  ✗ Stagnated at step {step}")
                break
        
        # Generate next code
        try:
            code = agent.step(task_description, code, feedback)
        except Exception as e:
            print(f"  ✗ Agent error: {e}")
            break
    
    # Get final statistics
    stats = auditor.get_statistics()
    agent_stats = agent.get_statistics()
    
    result = {
        'agent_type': agent_type,
        'obfuscation_level': obfuscation_level,
        'success': trajectory[-1]['loss'] == 0.0 if trajectory else False,
        'steps_to_solution': step if trajectory and trajectory[-1]['loss'] == 0.0 else max_steps,
        'final_loss': trajectory[-1]['loss'] if trajectory else 1.0,
        'oscillation_index': stats.get('oscillation_index', 0.0),
        'trajectory': trajectory,
        'auditor_stats': stats,
        'agent_stats': agent_stats
    }
    
    print(f"\n  Final: Success={result['success']}, OI={result['oscillation_index']:.2f}")
    
    return result


def run_stress_test(num_trials: int = 1):
    """
    Run the full stress test experiment.
    
    This is Experiment 1 from the paper.
    """
    print("=" * 70)
    print("STRESS TEST EXPERIMENT - ObfusCode vs Semantic Momentum")
    print("=" * 70)
    
    # Load configuration
    config = load_config()
    llm_kernel = LLMKernel(
        api_key=config['api_key'],
        base_url=config['base_url'],
        model=config['model']
    )
    
    # Create task
    task_description, clean_code, test_cases = create_simple_task()
    
    print(f"\nTask: Sum of even numbers")
    print(f"Test cases: {len(test_cases)}")
    print(f"Trials per configuration: {num_trials}")
    
    # Experiment configurations
    obfuscation_levels = [0, 1, 2]
    agent_types = ['baseline', 'momentum']
    
    all_results = []
    
    # Run experiments
    for level in obfuscation_levels:
        print(f"\n{'='*70}")
        print(f"OBFUSCATION LEVEL {level}")
        print('='*70)
        
        # Obfuscate the code
        if level == 0:
            obfuscated_code = clean_code
            print("Using clean code (no obfuscation)")
        else:
            obfuscated_code, mapping = obfuscate_code(clean_code, level=level)
            print(f"Obfuscated code ({len(mapping)} variables renamed):")
            print(obfuscated_code[:200] + "...")
        
        # Run trials for each agent type
        for agent_type in agent_types:
            for trial in range(num_trials):
                print(f"\n--- Trial {trial+1}/{num_trials} ---")
                
                result = run_single_trial(
                    agent_type=agent_type,
                    task_description=task_description,
                    initial_code=obfuscated_code,
                    test_cases=test_cases,
                    obfuscation_level=level,
                    llm_kernel=llm_kernel,
                    max_steps=15
                )
                
                result['trial'] = trial
                all_results.append(result)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputs/stress_test_{timestamp}.json"
    os.makedirs("outputs", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print('='*70)
    
    # Print summary
    print_summary(all_results)
    
    return all_results


def print_summary(results: List[Dict[str, Any]]):
    """Print experiment summary."""
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    # Group by configuration
    for level in [0, 1, 2]:
        print(f"\n[Level {level}]")
        for agent_type in ['baseline', 'momentum']:
            trials = [r for r in results if r['obfuscation_level'] == level and r['agent_type'] == agent_type]
            if not trials:
                continue
            
            success_rate = sum(r['success'] for r in trials) / len(trials) * 100
            avg_steps = sum(r['steps_to_solution'] for r in trials if r['success']) / max(sum(r['success'] for r in trials), 1)
            avg_oi = sum(r['oscillation_index'] for r in trials) / len(trials)
            
            print(f"  {agent_type.capitalize():10s}: Success={success_rate:5.1f}%, Steps={avg_steps:4.1f}, OI={avg_oi:5.2f}")
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    
    # Compare baseline vs momentum at each level
    for level in [0, 1, 2]:
        baseline_trials = [r for r in results if r['obfuscation_level'] == level and r['agent_type'] == 'baseline']
        momentum_trials = [r for r in results if r['obfuscation_level'] == level and r['agent_type'] == 'momentum']
        
        if baseline_trials and momentum_trials:
            baseline_success = sum(r['success'] for r in baseline_trials) / len(baseline_trials) * 100
            momentum_success = sum(r['success'] for r in momentum_trials) / len(momentum_trials) * 100
            
            baseline_oi = sum(r['oscillation_index'] for r in baseline_trials) / len(baseline_trials)
            momentum_oi = sum(r['oscillation_index'] for r in momentum_trials) / len(momentum_trials)
            
            print(f"\nLevel {level}:")
            print(f"  Success Rate: Baseline {baseline_success:.0f}% vs Momentum {momentum_success:.0f}%")
            print(f"  Oscillation:  Baseline {baseline_oi:.2f} vs Momentum {momentum_oi:.2f}")
            
            if momentum_success > baseline_success:
                improvement = momentum_success - baseline_success
                print(f"  → Momentum improves success by {improvement:.0f}%")
            
            if momentum_oi < baseline_oi:
                reduction = (baseline_oi - momentum_oi) / baseline_oi * 100
                print(f"  → Momentum reduces oscillation by {reduction:.0f}%")


if __name__ == '__main__':
    import sys
    
    # Parse arguments
    num_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    
    print(f"\nRunning stress test with {num_trials} trial(s) per configuration...")
    print("This will test both Baseline and Momentum agents at 3 obfuscation levels.")
    print("Total runs: 2 agents × 3 levels × {num_trials} trials = {2*3*num_trials} runs\n")
    
    try:
        results = run_stress_test(num_trials=num_trials)
        print("\n✓ Experiment completed successfully!")
    except Exception as e:
        print(f"\n✗ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
