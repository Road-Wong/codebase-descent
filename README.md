# Codebase Descent: 智能体软件工程的非凸优化理论框架

> 将LLM驱动的代码迭代开发形式化为优化问题，验证软件工程与深度学习训练的数学同构性

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 项目简介

当前LLM智能体在迭代开发中频繁陷入"反复修复、来回震荡"的困境，业界对此只有经验性描述而缺乏理论解释。**Codebase Descent** 将这一现象锚定在非凸优化的数学框架下，使智能体的震荡、停滞、发散获得了与SGD病态行为同构的严格刻画，将"调Prompt"的经验实践提升为可分析、可预测、可控制的动力学问题。

核心贡献——**Semantic Momentum Optimizer (SMO)**——证明了优化理论中的经典洞见可以跨越连续与离散的鸿沟，在自然语言空间中生效，为"如何赋予智能体惯性而非仅赋予记忆"提供了原则性方案。

### 核心理论映射

| 软件工程概念 | 优化理论概念 | 实现位置 |
|------------|------------|---------|
| 上下文窗口 K | 批量大小 B | `agent/memory_buffer.py` |
| Temperature T | 学习率 η | `agent/optimizer.py` |
| 思维链 | 梯度预处理 | `agent/llm_kernel.py` |
| 错误反馈 | 梯度信号 | `benchmarks/*.py` |
| 语义动量 | Momentum | `agent/momentum_agent.py` |
| 代码混淆 | 去语义化 (De-semanticization) | `benchmarks/obfuscator.py` |

### 研究假设

- **H1**: 上下文窗口抑制震荡 — 增大K时，Loss曲线方差显著减小
- **H2**: 温度相变 — 存在临界温度T_crit，收敛率骤降
- **H3**: 局部极小值逃逸 — 高温能逃离低温卡住的平台

---

## 系统架构

基于控制理论的四模块解耦设计：

```
codebase-descent/
├── benchmarks/              # [Plant] 被控对象 — Loss Landscape定义
│   ├── abstract_env.py      # 环境抽象接口
│   ├── env_convex.py        # 凸优化环境 (基准, 6测试用例)
│   ├── env_saddle.py        # 鞍点环境 (主要, 26测试用例)
│   ├── env_lru_cache.py     # LRU缓存环境 (8测试用例)
│   ├── env_graph.py         # 图算法环境 (8测试用例)
│   ├── env_order_discount.py# 电商折扣环境 (15测试用例)
│   └── obfuscator.py        # 去语义化代码混淆器 (ObfusCode-Bench)
│
├── agent/                   # [Controller] 控制器 — LLM优化器
│   ├── llm_kernel.py        # LLM API封装 + 思维链推理
│   ├── optimizer.py         # SGD / Adaptive / Annealing优化器
│   ├── momentum_agent.py    # Semantic Momentum Optimizer (SMO)
│   └── memory_buffer.py     # 上下文窗口管理 (Ring Buffer)
│
├── harness/                 # [Disturbance + Observer]
│   ├── injector.py          # 难度校准器 (4级扰动)
│   ├── auditor.py           # 动力学审计 (6种状态 + 震荡指数)
│   └── metrics.py           # 距离度量函数
│
├── experiments/             # 实验协议
│   ├── protocol_calibration.py  # 阶段I: 难度校准
│   ├── protocol_dynamics.py     # 阶段II: 动力学扫描
│   └── run_stress_test.py       # Baseline vs SMO 对比实验
│
├── analysis/                # 可视化分析
│   └── visualizer.py        # 生成相图和轨迹图
│
└── data/                    # 实验数据
    ├── trajectories/        # 优化轨迹 (JSONL)
    └── landscape_scans/     # Loss曲面扫描
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API

复制模板并填入你的API密钥：

```bash
cp config.txt.example config.txt
```

`config.txt` 格式：
```
https://api.xiaomimimo.com/v1
mimo-v2-flash
sk-your-api-key-here
```

### 3. 运行校准实验

```bash
cd experiments
python protocol_calibration.py --env saddle
```

### 4. 运行动力学扫描

```bash
python protocol_dynamics.py --env saddle --noise 2 --trials 5
```

### 5. 运行Baseline vs SMO对比

```bash
python run_stress_test.py --env saddle --noise 2 --trials 10
```

### 6. 生成可视化

```bash
cd ../analysis
python visualizer.py ../data/trajectories/dynamics_scan_*.json
```

---

## 核心方法：Semantic Momentum

传统的"上下文堆叠"策略仅相当于无动量的SGD，容易受到LLM不稳定反馈的干扰而陷入极限环。SMO在自然语言空间维护语义动量向量：

1. **语义梯度提取** — 从当前Diff与报错信息提取结构化的修改方向
2. **动量更新** — `m_{t+1} = β · Summarize(m_t) + (1-β) · g_t`，EMA平滑抑制高频震荡
3. **动量注入** — 将动量转化为Prompt约束，防止智能体回滚已有修改

---

## 实验结果

### 环境复杂度测试

| 环境 | 测试用例 | 代码长度 | 任务类型 | Greedy | Annealed | Sweet Spot |
|------|---------|---------|---------|--------|----------|-----------|
| Convex | 6 | 36 | 简单函数 | 1步 | 1步 | — |
| Saddle | 26 | 2124 | 解释器 | 1-3步 | 1步 | — |
| LRU Cache | 8 | 722 | 数据结构 | 1步 | 1步 | — |
| Graph | 8 | 1623 | 图算法 | 1步 | 1步 | — |
| Order Discount | 15 | 2019 | 业务逻辑 | 1步 | 1步 | — |

**关键发现**: 当任务具有完整语义时，模型能力溢出；ObfusCode-Bench通过去语义化制造信息不对称，才能暴露智能体的动力学缺陷。

### 核心洞察

1. **从结果正确性到过程可控性** — 评估智能体不能只看"能不能做对"，必须关注"做得是否稳定、是否可控"
2. **LLM能力评估至关重要** — 环境设计必须匹配模型能力，否则无法观察到有意义的动力学行为
3. **任务越难，动量越重要** — 在ObfusCode-Bench上，Baseline成功率暴跌而SMO保持稳定

---

## 技术栈

- **语言**: Python 3.8+
- **LLM API**: MiMo v2-flash (Xiaomi)
- **核心库**: openai, numpy, matplotlib, seaborn
- **架构**: 控制理论启发的四模块解耦设计

---

## 贡献

欢迎提出Issue和Pull Request！

主要贡献方向：
- 新的环境设计（尤其是高难度、去语义化任务）
- 扰动策略改进（诱导性错误、耦合破坏等）
- SMO动量参数调优
- 多模型适配（GPT、Claude、开源模型等）

---

## 许可证

MIT License

---

## 致谢

- MiMo API (Xiaomi) — LLM支持
- 实验框架设计参考了控制理论和优化理论的经典文献
