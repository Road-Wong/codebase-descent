# AGENTS.md

## Project

Codebase Descent — RL/control-theory framework that models LLM code iteration as non-convex optimization. Chinese docs, English code.

## External Datasets

`external_datasets/` contains 14 benchmark datasets (~70MB):
- **Repair**: QuixBugs (40), BugsInPy (502), SWE-bench (2294)
- **Generation**: HumanEval (164), MBPP (974), EvalPlus, BigCodeBench (1140), DS-1000 (1000), CodeContests (~10K), APPS (~10K)
- **Reference**: LiveCodeBench, Magicoder, CodeGeeX, OpenCodeInterpreter

See `困难任务设计.md` for adapter analysis and priority recommendations.

## Commands

```bash
pip install -r requirements.txt
cp config.txt.example config.txt   # 3 lines: base_url, model, api_key
```

### Tests (no LLM/API needed)

Three standalone scripts — **not** pytest. Run each directly:

```bash
python test_new_architecture.py    # 11 tests: types, env, agent, loop runner, diff
python test_protocol.py            # 8 tests: SEARCH/REPLACE protocol, CodeEnv
python test_sgd_smo.py             # 8 tests: SGD evaluator, DiffAgent, metrics, SMO
```

### Experiments (requires config.txt with valid API key)

```bash
python experiments/run_experiment.py --config configs/saddle_baseline.yaml
python experiments/run_experiment.py --env saddle --agent momentum --trials 3
python experiments/run_experiment.py --stress --env saddle --trials 5
```

No linter, formatter, or typecheck commands are configured.

## Architecture

```
Agent (agents/)  →  Action (SEARCH/REPLACE)  →  Environment (environments/)
    ↑                                                    |
    └──────────── Observation (single error) ←───────────┘
```

| Package | Role |
|---------|------|
| `core/` | Type contracts (Pydantic), Gymnasium env/agent base classes, protocol parser |
| `environments/` | Test environments (convex, saddle, lru_cache, graph, order_discount) + SGD evaluator + obfuscator |
| `agents/` | LLM kernel, DiffAgent base, SMO/Baseline momentum agents, optimizer, memory buffer |
| `harness/` | Loop runner (DI episode engine), config manager, logger, metrics, auditor, injector |
| `experiments/` | run_experiment.py entrypoint + legacy scripts |
| `configs/` | YAML experiment configs |

## Key quirks

- **config.txt format**: 3 plain lines (base_url, model, api_key) — not JSON/YAML.
- **Test scripts use `sys.path.insert(0, ...)`** — run from repo root.
- **Backward-compat shims**: `agent/` re-exports from `agents/`, `benchmarks/` re-exports from `environments/`. Prefer the canonical packages.
- **Protocol format**: Aider-style `<<<< SEARCH / ==== / >>>>` with fuzzy matching (`core/protocol.py`).
- **SGDEvaluator** shuffles tests, stops at first fail, returns ONE error — agent never sees test cases.
- **Output dir**: experiments write to `outputs/` (gitignored).
- **External datasets**: `external_datasets/` contains 14 benchmark datasets. QuixBugs is the lowest-cost adapter for SMO validation. See `困难任务设计.md` for details.
