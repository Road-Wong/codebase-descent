# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Codebase Descent** formalizes LLM-driven code iteration as a non-convex optimization problem. The Agent (optimizer) applies SEARCH/REPLACE patches to Code (model) under hidden test feedback (environment). The core innovation is the **Semantic Momentum Optimizer (SMO)**, which maintains a natural-language EMA momentum vector to stabilize optimization against oscillation.

Language: Chinese documentation, English code and module names.

## External Datasets

The `external_datasets/` directory contains 14 benchmark datasets (~70MB total) for validating SMO across diverse task types:

**Code Repair (directly applicable):**
- **QuixBugs** (40 algorithms, single-line bugs) — lowest adapter cost
- **BugsInPy** (502 bugs, 17 real Python projects) — real-world PR-level bugs
- **SWE-bench** (2294 GitHub issues) — real issue→PR fixes (requires `pip install`)

**Code Generation (adaptable to repair tasks):**
- **HumanEval** (164 problems), **MBPP** (974 problems), **EvalPlus** (extended tests)
- **BigCodeBench** (1140 tasks), **DS-1000** (1000 data science tasks)
- **CodeContests** (~10K), **APPS** (~10K) — competitive programming

**Reference/Tools:** LiveCodeBench, Magicoder, CodeGeeX, OpenCodeInterpreter

See `困难任务设计.md` sections 七 and 十 for detailed analysis and adapter recommendations.

## Commands

### Setup
```bash
pip install -r requirements.txt
cp config.txt.example config.txt   # 3 lines: base_url, model, api_key
```

### Run Tests (no LLM required)
```bash
python test_new_architecture.py    # 11 tests: types, env, agent, loop runner, diff
python test_protocol.py            # 8 tests: SEARCH/REPLACE protocol, CodeEnv
python test_sgd_smo.py             # 8 tests: SGD evaluator, DiffAgent, metrics, SMO
```

### Run Experiments (requires config.txt with valid API key)
```bash
# YAML-driven experiment
python experiments/run_experiment.py --config configs/saddle_baseline.yaml

# CLI quick run
python experiments/run_experiment.py --env saddle --agent momentum --trials 3

# Stress test: Baseline vs SMO at multiple obfuscation levels
python experiments/run_experiment.py --stress --env saddle --trials 5

# Legacy scripts (backward compatible)
python experiments/protocol_calibration.py --env saddle
python experiments/run_stress_test.py 1
```

## Architecture (RL Control Theory)

```
Agent (agents/)  →  Action (SEARCH/REPLACE)  →  Environment (environments/)
    ↑                                                    |
    └──────────── Observation (single error) ←───────────┘
```

### Core — `core/` (Type Contracts)

- **`types.py`**: Pydantic models — `State`, `Action`, `Observation`, `TrajectoryStep`, `EpisodeResult`, `ExperimentConfig`. All data crossing module boundaries must be typed.
- **`env_base.py`**: `AbstractEnvironment` (Gymnasium-style: `step() → (obs, reward, done, truncated, info)`). Concrete `CodeEnv` loads from `environments/assets/`.
- **`agent_base.py`**: `AbstractAgent` — `act(observation) → Action`.
- **`protocol.py`**: `SearchReplaceParser` — Aider-style `<<<< SEARCH / ==== / >>>>` format. `apply_patch()` with fuzzy matching. `ProtocolError` for violations.
- **`diff_utils.py`**: `generate_diff()`, `apply_diff()`, `make_diff_action()` — unified diff utilities.

### Environments — `environments/` (Plant)

- **`evaluator.py`**: `SGDEvaluator` — **random single-sample feedback**. Shuffles tests, stops at first fail, returns ONE error. Agent never sees test cases.
- **`env_convex.py`**: 6 tests, simple function.
- **`env_saddle.py`**: 26 tests, lexer+parser interpreter (main task).
- **`env_lru_cache.py`**: 8 tests, data structure.
- **`env_graph.py`**: 8 tests, graph algorithms.
- **`env_order_discount.py`**: 15 tests, business logic.
- **`obfuscator.py`**: AST-based code obfuscation (levels 0–3).
- **`assets/original_code.py`**: Ground truth c* for the Saddle task.

### Agents — `agents/` (Controller)

- **`base_agent.py`**: `DiffAgent` — base class enforcing SEARCH/REPLACE output. Validates format, previews patches. `RandomDiffAgent` for testing.
- **`momentum_agent.py`**: `SemanticMomentumOptimizer` (SMO) — 3-step loop:
  1. `_extract_gradient()` — LLM analyzes single error + last diff → g_t
  2. `_update_momentum()` — EMA: m_{t+1} = β·m_t + (1-β)·g_t
  3. `_generate_with_momentum()` — momentum injected as highest-priority system prompt constraint
  `BaselineAgent` — no-momentum control group (β=0).
- **`llm_kernel.py`**: `LLMKernel` — OpenAI-compatible API wrapper.
- **`optimizer.py`**: `CodeOptimizer` + `AdaptiveOptimizer` (legacy, for protocol_calibration).
- **`memory_buffer.py`**: Ring buffer for context window.

### Harness — `harness/` (Coordinator)

- **`loop_runner.py`**: `LoopRunner` — DI-driven episode engine. `run_episode()` orchestrates: agent.act → env.step → logger.record.
- **`config_manager.py`**: YAML → `ExperimentConfig`. `create_baseline_config()`, `create_smo_config()`.
- **`logger.py`**: `TrajectoryLogger` (JSONL per-step), `ExperimentLogger` (per-experiment directory).
- **`metrics.py`**: `CompositeMetric` — L_total = L_test + λ·L_ast. `ast_distance()` for structural divergence. `oscillation_index()` for limit cycle detection.
- **`auditor.py`**: `DynamicsAuditor` — 6-state classification (WARMUP/CONVERGING/OSCILLATING/STAGNATING/DIVERGING/SOLVED).
- **`injector.py`**: `NoiseInjector` — 4-level perturbation for difficulty calibration.

### Backward Compatibility

- `benchmarks/` → re-exports from `environments/`
- `agent/` → re-exports from `agents/`
- Old experiment scripts still work via `sys.path` manipulation

## Core Theory Mapping

| Software Engineering | Optimization Theory | Implementation |
|---|---|---|
| Single test feedback | SGD single-sample gradient | `environments/evaluator.py` |
| SEARCH/REPLACE patch | Bounded gradient step Δc | `core/protocol.py` |
| Context window K | Batch size B | `agents/memory_buffer.py` |
| Temperature T | Learning rate η | `agents/llm_kernel.py` |
| Semantic momentum m_t | EMA momentum | `agents/momentum_agent.py` |
| Code obfuscation | De-semanticization | `environments/obfuscator.py` |
| AST distance | Structural divergence | `harness/metrics.py` |
| Oscillation index | Limit cycle detection | `harness/metrics.py` |

## Key Anti-Cheating Mechanisms

1. **Blind box testing**: Agent never sees test cases. `SGDEvaluator` shuffles and stops at first fail.
2. **Bounded step size**: Agent must output SEARCH/REPLACE, not full code. Failed patches = wasted steps.
3. **Single error feedback**: Agent receives ONE error per step, not a global report.
4. **Momentum constraint**: SMO injects historical direction as system prompt, preventing oscillation.
5. **Information asymmetry**: Agent sees `Observation`, never `State` or ground truth.
