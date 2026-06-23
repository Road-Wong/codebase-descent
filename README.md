# Codebase Descent: LLM 代码优化的非凸优化理论框架

> 将 LLM 驱动的代码迭代开发形式化为强化学习问题，验证软件工程与深度学习训练的数学同构性

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 项目简介

当前 LLM 智能体在迭代开发中频繁陷入"反复修复、来回震荡"的困境，业界对此只有经验性描述而缺乏理论解释。**Codebase Descent** 将这一现象锚定在非凸优化的数学框架下：

- **Agent（优化器）** 对 **Code（模型）** 在 **Tests（环境）** 中进行 **梯度下降**
- 每次迭代：Agent 输出 SEARCH/REPLACE 局部补丁（Δc），环境返回单条随机报错（SGD 单样本梯度）
- **Semantic Momentum Optimizer (SMO)** 维护自然语言动量向量，抑制高频震荡

### 核心防作弊机制

| 机制 | 作用 | 实现 |
|------|------|------|
| 盲盒测试 | Agent 看不到测试用例 | `SGDEvaluator` 随机打乱，首个 Fail 即停 |
| 单样本反馈 | 每次只返回 1 条报错 | 模拟 SGD 随机梯度 |
| 局部 Diff 步长 | 不能全量重写代码 | `DiffAgent` 强制 SEARCH/REPLACE 格式 |
| 动量约束 | 不能随意推翻历史修复 | SMO 的 EMA 向量注入 System Prompt |

---

## 系统架构

```
Agent (agents/)  →  Action (SEARCH/REPLACE)  →  Environment (environments/)
    ↑                                                    |
    └──────────── Observation (单条报错) ←────────────────┘
```

```
codebase-descent/
├── core/                          # 类型契约与接口
│   ├── types.py                   # Pydantic: State, Action, Observation
│   ├── env_base.py                # Gymnasium 环境接口 + CodeEnv
│   ├── agent_base.py              # Agent 接口
│   ├── protocol.py                # SEARCH/REPLACE 协议解析与应用
│   └── diff_utils.py              # Unified diff 工具
│
├── environments/                  # [Plant] 被控对象
│   ├── evaluator.py               # SGD 随机单样本反馈测试器
│   ├── env_saddle.py              # 主任务: 表达式解析器 (26 tests)
│   ├── env_convex.py              # 基准: 简单函数 (6 tests)
│   ├── env_lru_cache.py           # LRU 缓存 (8 tests)
│   ├── env_graph.py               # 图算法 (8 tests)
│   ├── env_order_discount.py      # 电商折扣 (15 tests)
│   ├── obfuscator.py              # AST 代码混淆器
│   └── assets/original_code.py    # 全局最优解 c*
│
├── agents/                        # [Controller] 控制器
│   ├── base_agent.py              # DiffAgent 基类 (强制 SEARCH/REPLACE)
│   ├── momentum_agent.py          # SMO + BaselineAgent
│   ├── llm_kernel.py              # LLM API 封装
│   ├── optimizer.py               # SGD/Adaptive 优化器 (legacy)
│   └── memory_buffer.py           # 上下文窗口管理
│
├── harness/                       # [协调器]
│   ├── loop_runner.py             # RL Episode 执行引擎 (DI)
│   ├── config_manager.py          # YAML 配置管理
│   ├── logger.py                  # JSONL 轨迹记录
│   ├── metrics.py                 # L_test + L_ast 复合度量
│   ├── auditor.py                 # 动力学状态分类器
│   └── injector.py                # 难度校准器
│
├── external_datasets/             # 外部基准数据集 (14个, ~70MB)
│   ├── QuixBugs/                  # 40 算法, 单行 bug
│   ├── BugsInPy/                  # 502 bugs, 17 个真实项目
│   ├── SWE-bench/                 # 2294 GitHub issues
│   ├── human-eval/                # 164 题 (OpenAI)
│   ├── mbpp/                      # 974 题 (Google)
│   ├── evalplus/                  # HumanEval+/MBPP+ 扩展测试
│   ├── bigcodebench/              # 1140 题
│   ├── DS-1000/                   # 1000 数据科学题
│   ├── code_contests/             # ~10K 竞赛编程题
│   ├── apps/                      # ~10K 竞赛编程题
│   ├── LiveCodeBench/             # 实时代码生成基准
│   ├── magicoder/                 # OSS-Instruct 合成数据
│   ├── CodeGeeX/                  # 多语言代码生成
│   └── OpenCodeInterpreter/       # 代码生成+执行
│
├── configs/                       # YAML 实验配置
├── experiments/                   # 实验脚本
├── tests/                         # pytest 测试套件 (78 tests, no LLM required)
├── docs/                          # 文档
│   ├── design/                    # 设计文档 (困难任务设计.md 等)
│   ├── reports/                   # 历史实验报告 (gitignored)
│   └── notes/                     # 个人笔记 (gitignored)
└── analysis/                      # 可视化
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API

```bash
cp config.txt.example config.txt
# 编辑 config.txt: base_url, model, api_key
```

### 3. 运行测试（无需 API）

```bash
pytest tests/                      # 78 tests, no LLM required
pytest tests/ -v                   # verbose mode
pytest tests/ -k "test_types"      # run specific test
```

### 4. 运行实验

```bash
# YAML 配置驱动
python experiments/run_experiment.py --config configs/saddle_baseline.yaml

# CLI 快速运行
python experiments/run_experiment.py --env saddle --agent momentum --trials 3

# 压力测试
python experiments/run_experiment.py --stress --env saddle --trials 5
```

---

## 外部基准数据集

`external_datasets/` 目录包含 14 个基准数据集（~70MB），用于在多样化任务上验证 SMO 的有效性：

### 代码修复类（直接适用）

| 数据集 | 规模 | 特点 | 适配成本 |
|---|---|---|---|
| **QuixBugs** | 40 算法 | 单行 bug，教科书式缺陷 | 低 |
| **BugsInPy** | 502 bugs | 17 个真实 Python 项目的 PR 级 bug | 高 |
| **SWE-bench** | 2294 issues | 真实 GitHub issue → PR 修复 | 极高 |

### 代码生成类（可改编为修复任务）

| 数据集 | 规模 | 特点 | 适配成本 |
|---|---|---|---|
| **HumanEval** | 164 题 | OpenAI 标准基准 | 低 |
| **MBPP** | 974 题 | Google 基础编程题 | 低 |
| **DS-1000** | 1000 题 | 数据科学任务（7 个库） | 低 |
| **BigCodeBench** | 1140 题 | 实际编码任务 | 中 |
| **CodeContests** | ~10K 题 | 竞赛编程（DeepMind） | 高 |
| **APPS** | ~10K 题 | 竞赛编程（Hendrycks） | 高 |

### 推荐适配优先级

```
P1（强烈建议）：QuixBugs adapter — 100 行代码，低成本高回报
P2（锦上添花）：HumanEval/MBPP 改编 — 测试代码生成→修复的迁移能力
P3（可选）：BugsInPy 选 1-2 个轻量项目 — 真实世界验证
```

详见 `docs/design/困难任务设计.md` 第七至十节。

---

## 核心理论映射

| 软件工程概念 | 优化理论概念 | 实现位置 |
|------------|------------|---------|
| 单条测试反馈 | SGD 单样本梯度 | `environments/evaluator.py` |
| SEARCH/REPLACE 补丁 | 有界梯度步长 Δc | `core/protocol.py` |
| 上下文窗口 K | 批量大小 B | `agents/memory_buffer.py` |
| Temperature T | 学习率 η | `agents/llm_kernel.py` |
| 语义动量 m_t | EMA 动量 | `agents/momentum_agent.py` |
| 代码混淆 | 去语义化 | `environments/obfuscator.py` |
| AST 距离 | 结构散度 | `harness/metrics.py` |
| 震荡指数 | 极限环检测 | `harness/metrics.py` |

---

## SMO 算法

**Semantic Momentum Optimizer** 是核心创新。三步循环：

1. **梯度提取** — LLM 分析单条报错 + 上次 diff → g_t（语义梯度）
2. **动量更新** — EMA: `m_{t+1} = β · m_t + (1-β) · g_t`
3. **动量注入** — 将 m_{t+1} 作为 System Prompt 最高优先级约束

β 参数控制惯性：
- β = 0.0 → 纯 SGD（无记忆，易震荡）
- β = 0.7 → 标准动量（平衡）
- β = 0.9 → 高惯性（强平滑，反应慢）

---

## 技术栈

- **语言**: Python 3.8+
- **LLM API**: MiMo v2.5 (Xiaomi)
- **核心库**: openai, numpy, matplotlib, pydantic, pyyaml
- **架构**: Gymnasium-style RL + 控制理论启发

---

## 许可证

MIT License
