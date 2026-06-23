"""
Baseline vs SMO Comparison Experiment on Saddle Environment
============================================================

Experiment design:
  - Environment: SimpleInterpreter (26-expression arithmetic parser)
  - Initial state: BROKEN code (syntactically valid, semantically wrong)
  - Agent task: Fix the code via SEARCH/REPLACE patches
  - Feedback: SGD-style single error per step

Runs 10 trials each for BaselineAgent and SemanticMomentumOptimizer.

Outputs:
  outputs/comparison_<timestamp>.json   (raw trajectory data + codes)
  Then run:  python analysis/plot_figures.py outputs/comparison_<timestamp>.json
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.env_base import AbstractEnvironment
from core.types import Action, ActionFormat, Observation, State
from core.protocol import make_action, extract_protocol_block
from agents.llm_kernel import LLMKernel
from agents.momentum_agent import SemanticMomentumOptimizer, BaselineAgent
from agents.base_agent import DiffAgent
from harness.auditor import DynamicsAuditor


# =========================================================================
# Broken Initial Code — the starting point for the agent
# =========================================================================

BROKEN_CODE = '''class SimpleInterpreter:
    """A simple arithmetic expression interpreter."""
    def __init__(self):
        self.tokens = []

    def tokenize(self, expr):
        """Tokenize an arithmetic expression."""
        tokens = []
        i = 0
        while i < len(expr):
            if expr[i].isspace():
                i += 1
                continue
            if expr[i].isdigit():
                num = ''
                while i < len(expr) and expr[i].isdigit():
                    num += expr[i]
                    i += 1
                tokens.append(('NUM', int(num)))
            elif expr[i] in '+-*/':
                tokens.append(('OP', expr[i]))
                i += 1
            elif expr[i] in '()':
                tokens.append(('PAREN', expr[i]))
                i += 1
            else:
                i += 1
        return tokens

    def evaluate(self, expr):
        tokens = self.tokenize(expr)
        return self._eval_expr(tokens, 0)[0]

    def _eval_expr(self, tokens, pos):
        left, pos = self._eval_factor(tokens, pos)
        while pos < len(tokens) and tokens[pos][0] == 'OP' and tokens[pos][1] in '+-':
            op = tokens[pos][1]
            pos += 1
            right, pos = self._eval_factor(tokens, pos)
            if op == '+':
                left = left + right
            else:
                left = left - right
        return left, pos

    def _eval_term(self, tokens, pos):
        left, pos = self._eval_factor(tokens, pos)
        while pos < len(tokens) and tokens[pos][0] == 'OP' and tokens[pos][1] in '*/':
            op = tokens[pos][1]
            pos += 1
            right, pos = self._eval_factor(tokens, pos)
            if op == '*':
                left = left * right
            else:
                left = left // right
        return left, pos

    def _eval_factor(self, tokens, pos):
        if tokens[pos][0] == 'NUM':
            return tokens[pos][1], pos + 1
        elif tokens[pos][0] == 'PAREN' and tokens[pos][1] == '(':
            pos += 1
            result, pos = self._eval_expr(tokens, pos)
            pos += 1
            return result, pos
        return 0, pos
'''

# The bug: evaluate() calls _eval_expr, but _eval_expr calls _eval_factor
# instead of _eval_term. This breaks operator precedence:
#   '2+3*4' => (2+3)*4 = 20  (wrong, should be 14)
#   '(2+3)*4' => (2+3)*4 = 20  (correct by accident)
# The fix: _eval_expr should call _eval_term, not _eval_factor.


# =========================================================================
# Saddle Environment — custom for this experiment
# =========================================================================

class SaddleExperimentEnv(AbstractEnvironment):
    """
    Saddle environment for the comparison experiment.
    Starts with BROKEN code that the agent must fix.
    """

    TEST_CASES = [
        ("1+1", 2), ("2*3", 6), ("10-5", 5), ("8/2", 4),
        ("2+3*4", 14), ("(2+3)*4", 20), ("2*3+4*5", 26), ("5+3*2-4", 7),
        ("10-2-3", 5), ("1+2+3+4", 10), ("100/10/2", 5), ("(1+2)*(3+4)", 21),
        ("((1+2)*3)+4", 13), ("(1+(2*3))+4", 11), ("((2+3)*(4+5))", 45),
        ("1+2*3+4*5", 27), ("(1+2)*(3+4)*(5+6)", 231), ("10-5+3*2", 11),
        ("100/10+5*2", 20), ("0+0", 0), ("1*1", 1), ("10/10", 1),
        ("5-5", 0), ("100+200", 300), ("50*4", 200), ("1000/10", 100),
    ]

    def __init__(self, max_steps: int = 20):
        super().__init__(
            ground_truth_code=BROKEN_CODE,
            max_steps=max_steps,
            obfuscation_level=0,
        )

    def _get_task_description(self) -> str:
        return (
            "Implement a SimpleInterpreter class with methods:\n"
            "- tokenize(expr): tokenize arithmetic expression into (type, value) pairs\n"
            "- evaluate(expr): evaluate expression and return integer result\n"
            "- _eval_expr, _eval_term, _eval_factor: recursive descent parser helpers\n"
            "Support +, -, *, /, parentheses. Integer division for /. Operator precedence applies.\n"
            "Examples: '2+3*4'=14, '(2+3)*4'=20, '1+1'=2, '10/3'=3"
        )

    def _evaluate(self, code: str) -> dict:
        import ast
        passed = 0
        total = len(self.TEST_CASES)
        errors = []
        has_syntax_error = False

        try:
            ast.parse(code)
            namespace = {}
            exec(code, namespace)

            if "SimpleInterpreter" not in namespace:
                errors.append("Class 'SimpleInterpreter' not found")
                return {
                    "loss": 1.0, "passed_tests": 0, "total_tests": total,
                    "errors": errors, "has_syntax_error": True,
                }

            interpreter = namespace["SimpleInterpreter"]()
            for expr, expected in self.TEST_CASES:
                try:
                    result = interpreter.evaluate(expr)
                    if result == expected:
                        passed += 1
                    else:
                        errors.append(f"evaluate('{expr}') = {result}, expected {expected}")
                except Exception as e:
                    errors.append(f"Runtime error on '{expr}': {e}")

        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            has_syntax_error = True
        except Exception as e:
            errors.append(f"Execution error: {e}")
            has_syntax_error = True

        loss = 1.0 - (passed / total) if total > 0 else 1.0
        return {
            "loss": loss, "passed_tests": passed, "total_tests": total,
            "errors": errors, "has_syntax_error": has_syntax_error,
        }

    def _prepare_initial_code(self) -> str:
        return BROKEN_CODE


# =========================================================================
# LLM-Powered Agents
# =========================================================================

class LLMBaselineAgent(DiffAgent):
    """Baseline agent: LLM generates SEARCH/REPLACE patches, no momentum."""

    def __init__(self, llm_kernel: LLMKernel, temperature: float = 0.7):
        super().__init__()
        self.llm = llm_kernel
        self.temperature = temperature

    def _generate_patch(self, observation: Observation) -> str:
        error_msg = observation.errors[0] if observation.errors else "No error"

        prompt = f"""Fix this code. Output a SEARCH/REPLACE patch.

Code:
```python
{observation.current_code}
```

Error: {error_msg}

Format:
<<<< SEARCH
[exact lines from the code above]
====
[corrected lines]
>>>>
"""

        result = self.llm.generate_code(
            task_description=prompt, current_code="", error_feedback=None,
            temperature=self.temperature, max_tokens=1024, raw=True,
            system_prompt="You are a code optimization agent. You MUST output SEARCH/REPLACE patches in the exact format requested. Never output raw code blocks — always use <<<< SEARCH / ==== / >>>> markers. Output ONLY the patch, no explanations.",
        )
        result = extract_protocol_block(result.strip())
        if "<<<< SEARCH" not in result:
            if "```python" in result:
                start = result.find("```python") + 9
                end = result.find("```", start)
                if end != -1:
                    code = result[start:end].strip()
                    lines = observation.current_code.strip().split("\n")
                    search_line = lines[0] if lines else ""
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith("def ") or stripped.startswith("class "):
                            search_line = line
                            break
                    return make_action(search_line, code)
            lines = observation.current_code.strip().split("\n")
            return make_action(lines[0] if lines else "", lines[0] if lines else "")
        return result


class LLMSMOAgent(DiffAgent):
    """SMO agent: LLM with momentum as concise context in a single prompt."""

    def __init__(self, llm_kernel: LLMKernel, beta: float = 0.7, temperature: float = 0.7):
        super().__init__()
        self.llm = llm_kernel
        self.beta = beta
        self.temperature = temperature
        self.momentum_text = ""
        self.gradient_history = []
        self.diff_history = []

    def _extract_gradient(self, observation: Observation) -> str:
        error_msg = observation.errors[0] if observation.errors else "No error"
        last_diff = self.diff_history[-1] if self.diff_history else "None"

        prompt = f"""Analyze this error and extract a one-sentence fix direction.

CODE SNIPPET:
{observation.current_code[:800]}

ERROR: {error_msg}

LAST CHANGE: {last_diff[:300]}

Reply with ONE sentence: what to fix and why. Be specific about code locations."""

        gradient = self.llm.generate_code(
            task_description=prompt, current_code="", error_feedback=None,
            temperature=0.3, max_tokens=150,
        ).strip()
        self.gradient_history.append(gradient)
        return gradient

    def _update_momentum(self, gradient: str):
        if not self.momentum_text:
            self.momentum_text = gradient
            return

        prompt = f"""Combine these two optimization directions into ONE sentence.
Keep the consistent theme. Drop contradictions.

Previous (weight {self.beta}): {self.momentum_text}
New (weight {1-self.beta}): {gradient}

Combined direction:"""

        self.momentum_text = self.llm.generate_code(
            task_description=prompt, current_code="", error_feedback=None,
            temperature=0.4, max_tokens=150,
        ).strip()

    def _generate_patch(self, observation: Observation) -> str:
        gradient = self._extract_gradient(observation)
        self._update_momentum(gradient)

        error_msg = observation.errors[0] if observation.errors else "No error"
        mom_ctx = f"\nOptimization direction: {self.momentum_text}\n" if self.momentum_text else ""

        prompt = f"""Fix this code. Output a SEARCH/REPLACE patch.{mom_ctx}
Code:
```python
{observation.current_code}
```

Error: {error_msg}

Format:
<<<< SEARCH
[exact lines from the code above]
====
[corrected lines]
>>>>
"""

        result = self.llm.generate_code(
            task_description=prompt, current_code="", error_feedback=None,
            temperature=self.temperature, max_tokens=1024, raw=True,
            system_prompt="You are a code optimization agent. You MUST output SEARCH/REPLACE patches in the exact format requested. Never output raw code blocks — always use <<<< SEARCH / ==== / >>>> markers. Output ONLY the patch, no explanations.",
        ).strip()

        # Extract protocol block from response (strip any preamble/explanation)
        result = extract_protocol_block(result)

        self.diff_history.append(result)

        if "<<<< SEARCH" not in result:
            if "```python" in result:
                start = result.find("```python") + 9
                end = result.find("```", start)
                if end != -1:
                    code = result[start:end].strip()
                    lines = observation.current_code.strip().split("\n")
                    search_line = lines[0] if lines else ""
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith("def ") or stripped.startswith("class "):
                            search_line = line
                            break
                    return make_action(search_line, code)
            lines = observation.current_code.strip().split("\n")
            return make_action(lines[0] if lines else "", lines[0] if lines else "")
        return result

    def reset(self):
        super().reset()
        self.momentum_text = ""
        self.gradient_history = []
        self.diff_history = []


# =========================================================================
# Configuration
# =========================================================================

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.txt")
    with open(config_path, "r") as f:
        lines = f.read().strip().split("\n")
    return {"base_url": lines[0], "model": lines[1], "api_key": lines[2]}


# =========================================================================
# Experiment Runner
# =========================================================================

def run_single_trial(agent, env, auditor, max_steps=20):
    """Run one episode, collect trajectory."""
    obs = env.reset()
    trajectory = []
    codes = [obs.current_code]

    for step in range(max_steps):
        action = agent.act(obs)
        next_obs, reward, done, truncated, info = env.step(action)

        dynamics = auditor.step(next_obs.loss, next_obs.current_code)

        trajectory.append({
            "step": step,
            "loss": next_obs.loss,
            "passed_tests": next_obs.passed_tests,
            "total_tests": next_obs.total_tests,
            "dynamics": dynamics.value,
            "reward": reward,
        })
        codes.append(next_obs.current_code)

        obs = next_obs
        if done or truncated:
            break

    stats = auditor.get_statistics()
    return {
        "success": obs.loss == 0.0,
        "total_steps": len(trajectory),
        "final_loss": obs.loss,
        "min_loss": min(t["loss"] for t in trajectory) if trajectory else 1.0,
        "oscillation_index": stats.get("oscillation_index", 0.0),
        "trajectory": trajectory,
        "codes": codes,
    }


def run_experiment(trials=10, max_steps=20):
    """Run full Baseline vs SMO comparison."""
    config = load_config()
    llm = LLMKernel(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model"],
    )

    env = SaddleExperimentEnv(max_steps=max_steps)

    # Verify initial state
    obs = env.reset()
    print(f"Environment: Saddle (Expression Parser)")
    print(f"Initial loss: {obs.loss:.3f} ({obs.passed_tests}/{obs.total_tests} tests pass)")
    print(f"Trials per agent: {trials}")
    print(f"Max steps per trial: {max_steps}")
    print()

    results = {"baseline": [], "smo": []}

    for agent_type in ["baseline", "smo"]:
        print(f"{'='*60}")
        print(f"Running {agent_type.upper()} agent...")
        print(f"{'='*60}")

        for trial in range(trials):
            print(f"\n  Trial {trial+1}/{trials}...", end="", flush=True)

            auditor = DynamicsAuditor(window_size=5)

            if agent_type == "baseline":
                agent = LLMBaselineAgent(llm, temperature=0.7)
            else:
                agent = LLMSMOAgent(llm, beta=0.7, temperature=0.7)

            result = run_single_trial(agent, env, auditor, max_steps)
            results[agent_type].append(result)

            status = "SOLVED" if result["success"] else "FAILED"
            print(f" {status} | loss={result['final_loss']:.3f} | steps={result['total_steps']} | OI={result['oscillation_index']:.2f}")

    return results


# =========================================================================
# Summary
# =========================================================================

def print_summary(results):
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)

    for agent_type in ["baseline", "smo"]:
        data = results[agent_type]
        n = len(data)
        success_rate = sum(1 for d in data if d["success"]) / n * 100 if n else 0
        mean_steps = np.mean([d["total_steps"] for d in data])
        mean_oi = np.mean([d["oscillation_index"] for d in data])
        mean_final_loss = np.mean([d["final_loss"] for d in data])

        print(f"\n  [{agent_type.upper()}]")
        print(f"    Success Rate:        {success_rate:.1f}%")
        print(f"    Mean Steps:          {mean_steps:.1f}")
        print(f"    Mean Final Loss:     {mean_final_loss:.3f}")
        print(f"    Mean Oscillation I.: {mean_oi:.2f}")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_data = {}
    for agent_type in ["baseline", "smo"]:
        save_data[agent_type] = []
        for r in results[agent_type]:
            save_data[agent_type].append({
                "success": r["success"],
                "total_steps": r["total_steps"],
                "final_loss": r["final_loss"],
                "min_loss": r["min_loss"],
                "oscillation_index": r["oscillation_index"],
                "trajectory": r["trajectory"],
                "codes": r.get("codes", []),
            })

    json_path = os.path.join(output_dir, f"comparison_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Raw data saved to: {json_path}")

    return json_path, save_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    results = run_experiment(
        trials=args.trials,
        max_steps=args.steps,
    )
    json_path, save_data = print_summary(results)
