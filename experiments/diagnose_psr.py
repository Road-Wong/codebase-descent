"""
Diagnose Patch Success Rate — Step-by-step analysis.

Runs a single trial and prints every SEARCH/REPLACE block the agent
produces, whether it matched, and why it failed.

Usage:
    python experiments/diagnose_psr.py --agent baseline --noise 1
    python experiments/diagnose_psr.py --agent momentum --noise 1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.llm_kernel import LLMKernel
from agents.momentum_agent import BaselineAgent, SemanticMomentumOptimizer
from environments.env_saddle import SaddleEnvironment
from core.protocol import apply_patch, validate_action
from core.types import Action, ActionFormat


def load_config():
    with open('config.txt', 'r') as f:
        lines = f.read().strip().split('\n')
        return {'base_url': lines[0], 'model': lines[1], 'api_key': lines[2]}


def diagnose(agent_type: str, noise: int, max_steps: int = 10):
    config = load_config()
    llm = LLMKernel(api_key=config['api_key'], base_url=config['base_url'], model=config['model'])

    if agent_type == 'baseline':
        agent = BaselineAgent(llm, temperature=0.8)
    else:
        agent = SemanticMomentumOptimizer(llm, beta=0.6, temperature=0.8)

    env = SaddleEnvironment(obfuscation_level=noise, max_steps=max_steps)
    obs = env.reset()

    print(f"Agent: {agent_type} | Noise: {noise}")
    print(f"Initial loss: {obs.loss}")
    print(f"Code length: {len(obs.current_code)} chars")
    print(f"Code preview (first 300 chars):")
    print(obs.current_code[:300])
    print(f"{'='*70}")

    valid_count = 0
    invalid_count = 0

    for step in range(max_steps):
        print(f"\n--- Step {step} | Loss={obs.loss:.3f} ---")

        # Agent produces action
        action = agent.act(obs)

        # Validate format
        is_valid, err = validate_action(action.patch)
        print(f"  Format valid: {is_valid}")
        if not is_valid:
            print(f"  Format error: {err}")
            invalid_count += 1
        else:
            # Try to apply
            result = apply_patch(obs.current_code, action.patch)
            if result.success:
                print(f"  PATCH APPLIED OK")
                valid_count += 1
            else:
                print(f"  PATCH FAILED: {'; '.join(result.errors)}")
                invalid_count += 1

                # Show what SEARCH was looking for
                if "<<<< SEARCH" in action.patch:
                    parts = action.patch.split("====")
                    if len(parts) >= 1:
                        search_block = parts[0].replace("<<<< SEARCH", "").strip()
                        print(f"  SEARCH block ({len(search_block)} chars):")
                        for i, line in enumerate(search_block.split('\n')[:5]):
                            print(f"    {i}: {line}")
                        if len(search_block.split('\n')) > 5:
                            print(f"    ... ({len(search_block.split(chr(10)))} lines total)")

                # Show current code's first few lines for comparison
                code_lines = obs.current_code.strip().split('\n')
                print(f"  Current code first 5 lines:")
                for i, line in enumerate(code_lines[:5]):
                    print(f"    {i}: {line}")

        # Show the raw patch (truncated)
        patch_preview = action.patch[:200].replace('\n', '\\n')
        print(f"  Patch preview: {patch_preview}")

        # Step the environment
        obs, reward, done, truncated, info = env.step(action)
        print(f"  -> New loss: {obs.loss:.3f}, done={done}")

        if done:
            print(f"\n  SOLVED at step {step+1}!")
            break

    total = valid_count + invalid_count
    psr = valid_count / total if total > 0 else 0.0
    print(f"\n{'='*70}")
    print(f"RESULT: {valid_count}/{total} patches applied ({psr:.1%})")
    print(f"Final loss: {obs.loss:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', default='baseline', choices=['baseline', 'momentum'])
    parser.add_argument('--noise', type=int, default=1)
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()
    diagnose(args.agent, args.noise, args.steps)
