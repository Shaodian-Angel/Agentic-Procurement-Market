# Agentic Procurement Market Simulation

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![Architecture](https://img.shields.io/badge/Architecture-Orchestrator--MultiAgent-orange)
![Field](https://img.shields.io/badge/Field-Game%20Theory-green)

本项目旨在通过五个递进式的模拟场景（Simulations），测试并验证 Agentic Market 中的核心概念：**信念更新 (Belief Updating)**、**组合管理 (Portfolio Management)**、**双边适应 (Bilateral Adaptation)** 以及 **分布均衡 (Distributional Equilibrium)**。

> **核心目标：** 验证一个由 Orchestrator 管理的 Multi-agent 系统在不同市场复杂程度下，是否能收敛到稳定的分布均衡，并探讨决定均衡质量的关键因素。

---

## 1. 系统架构
系统采用层次化的 Orchestration 架构：
* **Orchestrator**: 负责全局策略与订单分配（Portfolio Management），管理下方各 Agent。
* **Negotiation Agent**: 执行具体的谈判逻辑，捕捉对手的反馈信号。
* **Risk Agent**: 负责风险评估与不确定性管理。
* **Assembly Agent**: 负责最终结果的整合。

## 2. 模拟计划 (Simulation Roadmap)

这种递进式设计是为了精确定位需要修正的特定组件。按顺序运行它们既是一种测试策略，也是一个研究计划。

| 阶段 | 场景名称 | 核心研究目标 |
| :--- | :--- | :--- |
| **Sim 1** | Passive Seller, 已知市场 | 建立 Baseline，验证信念更新能否正确识别 Seller 类型。 |
| **Sim 2** | Passive Seller, 未知市场 | 测试系统能否适应整体市场特征（二级信念更新）。 |
| **Sim 3** | Adaptive Seller, 简单规则 | 测量双边反馈循环的收缩率 $\lambda$ 与收敛性。 |
| **Sim 4** | Symmetric AI Seller | 测试对称博弈下的 Nash 均衡与协调失败情况。 |
| **Sim 5** | Asymmetric AI | 验证精细化买方对比集体信息卖方的实际价值。 |

## 3. 核心科学问题
1. 在什么条件下 Agentic Market 会收敛到稳定的分布均衡？
2. 双边反馈循环（Bilateral Feedback Loop）在何种参数下会收敛而非循环？
3. 个体买方的智能化优势能否抵消卖方的集体信息优势？

