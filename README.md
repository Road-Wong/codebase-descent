# Codebase Descent: 智能体软件工程的非凸优化理论框架

> 将LLM驱动的代码迭代开发形式化为优化问题,验证软件工程与深度学习训练的数学同构性

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Phase_1_Complete-yellow.svg)]()

---

## 📋 项目简介

**Codebase Descent**是一个实验框架,用于研究LLM智能体在代码优化过程中的动力学行为。我们将软件开发过程类比为神经网络训练,建立了软件工程概念与优化理论的映射关系。

### 核心理论映射

| 软件工程概念 | 优化理论概念 | 实现位置 |
|------------|------------|---------|
| 上下文窗口 K | 批量大小 B | `agent/memory_buffer.py` |
| Temperature T | 学习率 η | `agent/optimizer.py` |
| 思维链 | 梯度预处理 | `agent/llm_kernel.py` |
| 错误反馈 | 梯度信号 | `benchmarks/*.py` |

### 研究假设

- **H1**: 上下文窗口抑制震荡 - 增大K时,Loss曲线方差显著减小
- **H2**: 温度相变 - 存在临界温度T_crit,收敛率骤降
- **H3**: 局部极小值逃逸 - 高温能逃离低温卡住的平台

---

## 🏗️ 系统架构

基于控制理论的四模块解耦设计:

```
codebase-descent/
├── benchmarks/          # [Plant] 被控对象 - Loss Landscape定义
│   ├── abstract_env.py  # 环境抽象接口
│   ├── env_convex.py    # 凸优化环境(基准)
│   └── env_saddle.py    # 鞍点环境(主要,26测试用例)
│
├── agent/               # [Controller] 控制器 - LLM优化器
│   ├── llm_kernel.py    # MiMo API封装
│   ├── optimizer.py     # SGD/Adaptive优化器
│   └── memory_buffer.py # 上下文窗口管理
│
├── harness/             # [Disturbance + Observer]
│   ├── injector.py      # 难度校准器(4级扰动)
│   ├── auditor.py       # 动力学审计(5种状态)
│   └── metrics.py       # 距离度量函数
│
├── experiments/         # 实验协议
│   ├── protocol_calibration.py  # 阶段I: 难度校准
│   └── protocol_dynamics.py     # 阶段II: 动力学扫描
│
├── analysis/            # 可视化分析
│   └── visualizer.py    # 生成相图和轨迹图
│
└── data/                # 实验数据
    ├── trajectories/    # 优化轨迹(JSONL)
    └── landscape_scans/ # Loss曲面扫描
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API

确保 `config.txt` 包含:
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

### 5. 生成可视化

```bash
cd ../analysis
python visualizer.py ../data/trajectories/dynamics_scan_*.json
```

---

## 📊 当前进展

### ✅ 已完成 (Phase 1 & 2)

- [x] 完整的四模块架构实现
- [x] 5个环境实现:
  - ConvexEnvironment (6测试用例)
  - SaddleEnvironment增强版 (26测试用例)
  - LRUCacheEnvironment (8测试用例)
  - GraphEnvironment (8测试用例)
  - OrderDiscountEnvironment (15测试用例)
- [x] 4级难度扰动策略
- [x] 动力学审计系统
- [x] 实验协议和可视化工具
- [x] 完整的文档体系
- [x] 系统化测试多种任务类型

### 🔍 核心发现

**未找到Sweet Spot**: mimo-v2-flash对所有测试任务都能在1-3步内解决

- 经典算法任务: 太熟悉
- 业务逻辑任务: 理解能力强
- 根本原因: 模型能力 > 任务难度

### 🎯 建议方向

- [ ] 改变研究目标: 分析收敛质量、路径差异、稳定性
- [ ] 或寻找更弱的模型进行实验
- [ ] 或设计极端复杂的任务

### 📅 计划中 (Phase 3)

- [ ] 验证三个核心假设
- [ ] 多模型对比实验
- [ ] 论文撰写和投稿
- [ ] 开源发布

---

## 🔬 实验结果

### 环境复杂度测试

| 环境 | 测试用例 | 代码长度 | 任务类型 | Greedy | Annealed | Sweet Spot |
|------|---------|---------|---------|--------|----------|-----------|
| Convex | 6 | 36 | 简单函数 | ✓ 1步 | ✓ 1步 | ✗ |
| Saddle | 26 | 2124 | 解释器 | ✓ 1-3步 | ✓ 1步 | ✗ |
| LRU Cache | 8 | 722 | 数据结构 | ✓ 1步 | ✓ 1步 | ✗ |
| Graph | 8 | 1623 | 图算法 | ✓ 1步 | ✓ 1步 | ✗ |
| Order Discount | 15 | 2019 | 业务逻辑 | ✓ 1步 | ✓ 1步 | ✗ |

**发现**: mimo-v2-flash对所有任务类型都能快速解决，包括经典算法和业务逻辑。

### 核心洞察

1. **LLM能力评估至关重要** - 环境设计必须匹配模型能力
2. **任务类型影响显著** - 常见任务容易被快速解决
3. **扰动策略需优化** - 诱导性错误比破坏性错误更有效

---

## 📖 文档

- **[实验框架.md](实验框架.md)** - 完整的实验设计文档
- **[实验报告.md](实验报告.md)** - Phase 1阶段性总结
- **[Phase2进展报告.md](Phase2进展报告.md)** - Phase 2最终总结和建议 ⭐
- **[论文大纲.md](论文大纲.md)** - 理论框架和论文结构
- **[作者心得.md](作者心得.md)** - 开发心得

---

## 🛠️ 技术栈

- **语言**: Python 3.8+
- **LLM API**: MiMo v2-flash (Xiaomi)
- **核心库**: openai, numpy, matplotlib, seaborn
- **架构**: 模块化解耦设计

---

## 📈 研究方向建议

### 推荐方案: 改变研究目标

**接受现状，研究其他有价值的问题**:

1. **收敛质量分析**
   - 不同温度下代码的可读性和效率
   - 是否使用最佳算法

2. **路径依赖研究**
   - 贪婪 vs 退火的中间状态差异
   - 温度对探索策略的影响

3. **稳定性和鲁棒性**
   - 多次运行的一致性
   - 对不同扰动的敏感度

4. **上下文窗口效应**
   - K对收敛速度的影响
   - 验证H1假设的弱化版本

### 备选方案

1. **寻找更弱的模型**: GPT-3.5, CodeLlama-7B等
2. **极端复杂任务**: 多文件项目、并发编程等
3. **诱导性错误**: Level 5扰动策略

详见 **[Phase2进展报告.md](Phase2进展报告.md)**

---

## 🤝 贡献

欢迎提出Issue和Pull Request!

主要贡献方向:
- 新的环境设计
- 扰动策略改进
- 可视化工具增强
- 文档完善

---

## 📄 许可证

MIT License

---

## 📮 联系方式

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **讨论**: [GitHub Discussions](https://github.com/your-repo/discussions)

---

## 🙏 致谢

- MiMo API (Xiaomi) - LLM支持
- 实验框架设计参考了控制理论和优化理论的经典文献

---

**最后更新**: 2025年12月29日  
**项目状态**: Phase 1 完成,进入 Phase 2  
**版本**: v1.0
