# 贝叶斯验证度量 (Bayesian Validation Metric - BVM)

基于 NumPy 实现的贝叶斯验证度量方法，用于模型校准、验证和不确定性量化。

## 目录

- [概述](#概述)
- [算法原理](#算法原理)
- [BVM 与 TMCMC 的关系](#bvm-与-tmcmc-的关系)
- [核心特性](#核心特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [算例](#算例)
- [API 参考](#api-参考)
- [参考文献](#参考文献)

---

## 概述

贝叶斯验证度量 (Bayesian Validation Metric, BVM) 是一种广义贝叶斯方法，专为模型校准和验证而设计。与传统的高斯似然不同，BVM 通过用户定义的"一致性容差" (epsilon) 来量化模型预测与观测数据的匹配程度。

本实现提供了一个完整的框架，适用于：
- **模型校准** (Calibration)：基于观测数据推断模型参数
- **模型验证** (Validation)：评估模型预测能力和可靠性
- **不确定性量化**：考虑观测噪声和模型不确定性
- **模型比较与选择**：通过一致性概率比较不同模型

---

## 算法原理

### 数学基础

#### 1. BVM 似然函数

传统贝叶斯推断使用高斯似然：

```
L_Gaussian(θ) = ∏ᵢ exp(-0.5 * ((yᵢ - f(xᵢ, θ)) / σᵢ)²)
```

BVM 采用**容差带似然** (Tolerance Band Likelihood)：

```
L_BVM(θ) = ∏ᵢ P(|f(xᵢ, θ) - yᵢ| ≤ ε | σᵢ)
```

其中：
- **θ**: 模型参数
- **f(x, θ)**: 计算模型（Forward Model）
- **yᵢ**: 第 i 个观测输出
- **xᵢ**: 第 i 个观测输入
- **σᵢ**: 观测噪声标准差
- **ε (epsilon)**: 用户定义的一致性容差

---

#### 2. 一致性概率计算

假设观测数据服从正态分布：**yᵢ ~ N(ȳᵢ, σᵢ²)**，则预测值 **f(xᵢ, θ)** 落在容差带内的概率为：

```
Pᵢ(θ) = P(|f(xᵢ, θ) - yᵢ| ≤ ε)
      = P(yᵢ ∈ [f(xᵢ, θ) - ε, f(xᵢ, θ) + ε])
      = Φ((f(xᵢ, θ) - ȳᵢ + ε) / σᵢ) - Φ((f(xᵢ, θ) - ȳᵢ - ε) / σᵢ)
```

其中 **Φ(·)** 是标准正态分布的累积分布函数 (CDF)。

**物理意义：**
- 当 **f(xᵢ, θ) ≈ ȳᵢ** 时，Pᵢ(θ) 接近最大值
- 当 **|f(xᵢ, θ) - ȳᵢ| > ε + 3σᵢ** 时，Pᵢ(θ) 接近 0
- ε 越大，容忍度越高，似然越"平滑"
- ε 越小，要求越严格，似然越"尖锐"

---

#### 3. 对数似然函数

为避免数值下溢，在对数空间计算：

```
log L_BVM(θ) = Σᵢ log Pᵢ(θ)
```

**数值稳定性处理：**
- 当 Pᵢ(θ) → 0 时，添加小常数：`Pᵢ = max(Pᵢ, 1e-300)`
- 避免 `log(0)` 导致的 NaN 错误

---

#### 4. BVM 后验分布

根据贝叶斯公式，后验分布为：

```
p(θ | Data) ∝ p(θ) × L_BVM(θ)
```

其中：
- **p(θ)**: 先验分布（反映先验知识或工程经验）
- **L_BVM(θ)**: BVM 似然函数（反映数据一致性）

**后验采样：** 由于后验分布通常没有解析形式，需要使用 MCMC 方法（如 TMCMC）进行采样。

---

### BVM 的关键优势

#### 1. 灵活的容差控制
- **传统方法**: 硬编码高斯误差模型，假设误差服从正态分布
- **BVM**: 用户通过 ε 直接控制"可接受的误差范围"
- **优势**: 更符合工程实践（例如：结构位移误差在 ±5mm 内可接受）

#### 2. 鲁棒性
- **传统方法**: 对异常值敏感（高斯似然的指数惩罚）
- **BVM**: 只要预测在容差带内，似然不随误差大小显著变化
- **优势**: 对测量误差和模型偏差更鲁棒

#### 3. 物理可解释性
- **ε 的物理意义**: 直接对应工程容差或测量精度
  - 例如：结构工程中，位移测量精度为 ±2mm，可设 ε = 2mm
  - 例如：温度测量精度为 ±0.5°C，可设 ε = 0.5°C
- **后验样本的意义**: 所有使模型预测在容差范围内的参数组合

#### 4. 适用于模型不匹配
- **传统方法**: 假设模型完美，所有误差来自噪声
- **BVM**: 允许模型偏差（Model Discrepancy），只要在 ε 内即可
- **优势**: 适用于简化模型或现象学模型

---

### BVM 验证模式

BVM 提供两种验证模式来评估模型质量：

#### 模式 1: Strict Agreement (严格一致性)

**定义：** 一个参数 θ "一致"当且仅当它对**所有观测点**都在容差范围内。

**数学表达：**
```
P_strict(θ agrees) = ∏ᵢ Pᵢ(θ)
BVM_strict = E_θ[P_strict(θ)] = (1/N) Σⱼ ∏ᵢ Pᵢ(θⱼ)
```

**物理意义：**
- 回答问题："整条预测曲线与数据一致的概率是多少？"
- 用于**模型选择**：分数越高，模型越好
- 分数通常较低（因为要求所有点同时满足）

**示例：**
- 若有 10 个观测点，每点 Pᵢ = 0.9，则 P_strict = 0.9¹⁰ ≈ 0.35
- 若有 1 个点 Pᵢ = 0.1，其余 Pᵢ = 0.9，则 P_strict ≈ 0.039

---

#### 模式 2: Reliability (可靠性)

**定义：** 平均而言，每个观测点的一致性概率。

**数学表达：**
```
BVM_reliability = E_θ,i[Pᵢ(θ)] = (1/(N×M)) ΣⱼΣᵢ Pᵢ(θⱼ)
```

其中 N 是样本数，M 是观测点数。

**物理意义：**
- 回答问题："随机挑一个参数和一个观测点，一致的概率是多少？"
- 用于**模型可靠性评估**：分数越高，模型越可靠
- 分数通常较高（不要求全部点同时满足）

**示例：**
- 若有 10 个观测点，平均 Pᵢ = 0.85，则 BVM_reliability ≈ 0.85
- 即"平均有 85% 的概率预测正确"

---

### ε (Epsilon) 的选择

ε 是 BVM 最重要的超参数，决定了校准的严格程度。

#### 推荐准则：

| 情况 | 推荐 ε | 理由 |
|------|--------|------|
| **高精度实验数据** | ε = 1-2σ | 严格匹配数据 |
| **工程测量数据** | ε = 工程容差 | 符合工程规范 |
| **模型有偏差** | ε = 2-5σ | 容忍模型不匹配 |
| **探索性分析** | 逐步减小 ε | 观察后验收缩 |

#### 经验法则：

1. **从大到小测试**：
   ```python
   epsilon_list = [0.1, 0.03, 0.01]  # 逐步减小
   ```
   - 观察后验分布如何收缩
   - 找到"合理但不过拟合"的 ε

2. **相对误差法**：
   ```python
   epsilon = 0.05 * np.mean(np.abs(y_data))  # 5% 相对误差
   ```

3. **噪声水平法**：
   ```python
   epsilon = 2.0 * np.std(y_data)  # 2 倍数据标准差
   ```

---

## BVM 与 TMCMC 的关系

### 依赖关系图

```
┌─────────────────────────────────────────────────────────┐
│                    BVM 工作流程                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Step 1: 定义问题                                        │
│  ┌──────────────────────────────────────────┐          │
│  │ • 计算模型 f(x, θ)                       │          │
│  │ • 观测数据 {xᵢ, yᵢ, σᵢ}                 │          │
│  │ • 一致性容差 ε                            │          │
│  └──────────────────────────────────────────┘          │
│                      │                                   │
│                      ▼                                   │
│  Step 2: 构造 BVM 似然函数                              │
│  ┌──────────────────────────────────────────┐          │
│  │  BVMLikelihood(                          │          │
│  │    model=f,                              │  ◄─── BVM 核心
│  │    observed_input=x,                     │          │
│  │    observed_output_mean=y,               │          │
│  │    observed_output_std=σ,                │          │
│  │    epsilon=ε                             │          │
│  │  )                                       │          │
│  └──────────────────────────────────────────┘          │
│                      │                                   │
│                      ▼                                   │
│  Step 3: 定义先验分布                                   │
│  ┌──────────────────────────────────────────┐          │
│  │  prior = JointPrior([...])               │          │
│  │  init_samples = prior.sample(N)          │          │
│  └──────────────────────────────────────────┘          │
│                      │                                   │
│                      ▼                                   │
│  Step 4: 使用 TMCMC 采样后验分布                        │
│  ┌──────────────────────────────────────────┐          │
│  │  tmcmc = TransitionalMCMC(...)           │  ◄─── TMCMC 引擎
│  │  posterior = tmcmc.sample(               │          │
│  │    likelihood=bvm_lik,    ◄─┐           │          │
│  │    prior=prior,              │           │          │
│  │    init_samples=init_samples │           │          │
│  │  )                           │           │          │
│  └──────────────────────────────│───────────┘          │
│                                  │                       │
│  ┌───────────────────────────────┘                      │
│  │  依赖 TMCMC 的原因:                                  │
│  │  • BVM 似然函数无解析后验                            │
│  │  • 需要 MCMC 方法进行采样                            │
│  │  • TMCMC 自适应 + 无需 burn-in                       │
│  └────────────────────────────────────────────          │
│                      │                                   │
│                      ▼                                   │
│  Step 5: 验证模型质量                                   │
│  ┌──────────────────────────────────────────┐          │
│  │  validator = BVMValidator(...)           │  ◄─── BVM 验证
│  │  strict_score = validator.compute_bvm_score(│      │
│  │    theta_samples=posterior,              │          │
│  │    epsilon=ε,                            │          │
│  │    mode='strict'                         │          │
│  │  )                                       │          │
│  └──────────────────────────────────────────┘          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

### 为什么 BVM 依赖 TMCMC？

#### 1. 后验分布无解析形式

BVM 似然函数 `L_BVM(θ)` 通常非常复杂：
- 涉及非线性模型 f(x, θ)
- 多个观测点的乘积
- 正态 CDF 的嵌套计算

**结果：** 后验分布 `p(θ | Data) ∝ p(θ) × L_BVM(θ)` 无法写成简单的数学形式，无法直接采样。

---

#### 2. 需要 MCMC 采样

唯一方法是使用 **马尔可夫链蒙特卡洛 (MCMC)** 方法：
- 构造一条马尔可夫链，其平衡分布为后验分布
- 通过链的游走，生成后验样本

**常见 MCMC 方法：**
- Metropolis-Hastings (MH)
- Gibbs Sampling
- Hamiltonian Monte Carlo (HMC)
- **Transitional MCMC (TMCMC)** ← BVM 使用此方法

---

#### 3. 为什么选择 TMCMC？

| 特性 | 标准 MCMC | TMCMC | BVM 需求 |
|------|-----------|-------|----------|
| **需要 Burn-in** | ✓ (浪费样本) | ✗ (无需丢弃) | 高效采样 ✓ |
| **多峰探索** | ✗ (易陷入局部) | ✓ (退火策略) | BVM 似然可能多峰 ✓ |
| **自动调参** | ✗ (需手动调) | ✓ (自适应) | 用户友好 ✓ |
| **高维效率** | △ (收敛慢) | ✓ (重采样机制) | 处理多参数 ✓ |
| **收敛监控** | 困难 | 简单 (φ 进度) | 可视化友好 ✓ |

**结论：** TMCMC 完美适配 BVM 的需求：
- BVM 定义"什么是好参数"（通过似然函数）
- TMCMC 找到"这些好参数在哪里"（通过采样）

---

#### 4. TMCMC 在 BVM 中的具体作用

**输入给 TMCMC:**
```python
likelihood = BVMLikelihood(...)  # BVM 定义的似然
prior = JointPrior(...)         # 先验分布
init_samples = prior.sample(N)  # 从先验采样初始点
```

**TMCMC 内部流程:**
1. **阶段 0 (φ=0)**: 样本分布在整个先验空间
   - 探索所有可能的参数区域
   - 避免过早收敛到局部最优

2. **阶段 1-k (0<φ<1)**: 逐步增加 BVM 似然的影响
   - 权重机制：`w ∝ L_BVM^φ`
   - BVM 似然高的区域权重变大
   - 重采样聚焦到"一致性好"的参数

3. **阶段 k+1 (φ=1)**: 完全后验分布
   - 样本代表 `p(θ | Data) ∝ p(θ) × L_BVM(θ)`
   - 这些参数既符合先验知识，又匹配观测数据

**输出后验样本:**
```python
posterior = tmcmc.sample(...)  # Shape: (N, n_dim)
# 每一行是一个参数组合 θ
# 这些 θ 都使得模型预测在 ε 容差内
```

---

#### 5. 依赖关系总结

```
BVM 似然函数 (BVMLikelihood)
    │
    ├─ 定义了 log_likelihood(θ) 方法
    │  └─ 计算 Σᵢ log Pᵢ(θ)
    │
    ▼
传递给 TMCMC 采样器
    │
    ├─ TMCMC 调用 log_likelihood(θ) 来评估参数好坏
    │  └─ 用于计算权重、MH 接受率等
    │
    ▼
TMCMC 输出后验样本
    │
    ├─ 这些样本可用于:
    │  • 参数估计 (mean, std, quantiles)
    │  • 预测不确定性量化
    │  • 模型验证 (BVMValidator)
    │
    ▼
BVM 验证器 (BVMValidator)
    └─ 使用后验样本计算一致性概率
       • Strict: 整体一致性
       • Reliability: 平均可靠性
```

**关键点：**
- **BVM 不能独立工作**，必须结合 MCMC 方法
- **TMCMC 是 BVM 的采样引擎**，负责从后验分布生成样本
- **BVM 似然 + TMCMC 采样 = 完整的贝叶斯校准流程**

---

## 核心特性

### BVM 似然函数 (`BVMLikelihood`)
- 基于容差带的一致性概率计算
- 支持噪声数据（通过 `observed_output_std`）
- 支持确定性数据（σ → 0 的极限情况）
- 数值稳定的对数似然计算
- 批量化预测，支持向量化计算

### BVM 验证器 (`BVMValidator`)
- **Strict 模式**: 评估整体模型一致性
- **Reliability 模式**: 评估逐点可靠性
- 独立于校准过程，可用于验证集
- 自动生成验证报告

### 与 TMCMC 无缝集成
- BVM 似然函数符合 `LikelihoodFunction` 接口
- 自动处理形状转换和批量计算
- 支持 TMCMC 的所有采样策略（Gaussian/GMM 提议）

---

## 安装

### 依赖要求

```bash
numpy >= 1.20
scipy >= 1.7
matplotlib >= 3.3
```

BVM 模块依赖 TMCMC 模块，请确保已安装：
```bash
# 从父目录运行
cd BayesCalibration
```

---

## 快速开始

### 基础示例

```python
import numpy as np
from BayesCalibration.TMCMC.utils import PriorDistribution, JointPrior
from BayesCalibration.TMCMC.tmcmc import TransitionalMCMC
from BayesCalibration.BVM.likelihood import BVMLikelihood
from BayesCalibration.BVM.validator import BVMValidator

# ========================================
# 1. 定义计算模型
# ========================================
def my_model(input_concat):
    """
    模型函数
    input_concat: shape (n, x_dim + theta_dim)
    返回: shape (n,) 预测值
    """
    x = input_concat[:, 0]         # 输入变量
    a = input_concat[:, 1]         # 参数 1
    b = input_concat[:, 2]         # 参数 2
    return a * x + b               # 线性模型 y = ax + b

# ========================================
# 2. 准备观测数据
# ========================================
# 真实参数: a=2.0, b=1.0
x_obs = np.linspace(0, 10, 20).reshape(-1, 1)
y_true = 2.0 * x_obs.flatten() + 1.0
y_obs = y_true + np.random.normal(0, 0.5, size=20)  # 加噪声
y_std = np.full(20, 0.5)  # 噪声标准差

# ========================================
# 3. 定义先验分布
# ========================================
prior = JointPrior([
    PriorDistribution('uniform', low=0.0, high=5.0),   # a 的先验
    PriorDistribution('uniform', low=-2.0, high=4.0)   # b 的先验
])

# ========================================
# 4. 创建 BVM 似然函数
# ========================================
epsilon = 1.0  # 一致性容差（根据问题调整）

bvm_likelihood = BVMLikelihood(
    observed_input=x_obs,
    observed_output_mean=y_obs,
    observed_output_std=y_std,
    model=my_model,
    epsilon=epsilon
)

# ========================================
# 5. 使用 TMCMC 进行后验采样
# ========================================
init_samples = prior.sample(2000)

tmcmc = TransitionalMCMC(
    initial_beta=0.2,
    target_cov=1.0,
    adapt_beta=True,
    proposal_type='gaussian',
    verbose=False
)

posterior_samples = tmcmc.sample(
    likelihood=bvm_likelihood,
    prior=prior,
    init_samples=init_samples,
    n_mh_steps=20
)

# ========================================
# 6. 分析后验结果
# ========================================
print("后验统计:")
print(f"  a: {np.mean(posterior_samples[:, 0]):.3f} ± {np.std(posterior_samples[:, 0]):.3f}")
print(f"  b: {np.mean(posterior_samples[:, 1]):.3f} ± {np.std(posterior_samples[:, 1]):.3f}")

# ========================================
# 7. 模型验证
# ========================================
validator = BVMValidator(
    model_func=my_model,
    val_input=x_obs,
    val_output_mean=y_obs,
    val_output_std=y_std
)

validator.report(posterior_samples, epsilon=epsilon)
```

**预期输出：**
```
T-MCMC 采样: 2000 条链 × 2 维参数
...
✓ 完成! 总阶段数: 5 | 平均接受率: 45.2%

后验统计:
  a: 2.015 ± 0.084
  b: 0.957 ± 0.312

--- BVM Validation Report (eps=1.0) ---
Strict Agreement Probability (Model Testing): 0.892341
Reliability Metric (Average Pointwise):       0.976543
```

---

## 算例

我们提供了三个经过精心设计的算例，展示 BVM 在不同类型问题上的应用。

### 算例 1: Bacterial Growth Model (细菌生长模型)

**问题描述：**

Monod 方程描述细菌生长速率与底物浓度的关系：

```
μ(S, α₁, α₂) = α₁ × S / (α₂ + S)
```

其中：
- **S**: 底物浓度 (Substrate Concentration)
- **α₁**: 最大生长速率
- **α₂**: 半饱和常数 (Monod constant)

**数据来源：** Contois (1959) 的实验数据

**先验分布：**
- α₁ ~ N(0.17, 0.025²)
- α₂ ~ N(47.5, 3.0²)

**测试 ε 值：** [0.1, 0.03, 0.01]

**物理意义：**
- ε = 0.1: 容忍 10% 相对误差（宽松）
- ε = 0.01: 要求 1% 相对误差（严格）

**运行：**
```python
from BVM.main import run_example_1
run_example_1()
```

**输出文件：**
- `results/ex1_prior.png`: 先验预测包络
- `results/ex1_eps_0.1_posterior.png`: ε=0.1 的后验预测
- `results/ex1_eps_0.1_params.png`: 参数后验分布

**预期结果：**
- ε 减小时，后验分布逐渐收缩
- 后验均值收敛到真实参数附近
- Strict 一致性概率随 ε 增大而提高

---

### 算例 2: Toy Model (合成测试模型)

**问题描述：**

一个复杂的六参数非线性模型：

```
y(x, θ) = a₁×exp(-a₂×x)×sin(a₃×x) + a₄×log(1+a₅×x) + a₆×x²
```

**特点：**
- 高度非线性
- 参数耦合
- 多峰似然（测试 TMCMC 的多峰探索能力）

**数据生成：**
- 50 个观测点，x ∈ [0, 3]
- 加入 5% 高斯噪声

**先验分布：**
- a₁, a₂, a₃, a₅, a₆ ~ N(1.0, 0.3²)
- a₄ ~ N(10.0, 0.3²)

**测试 ε 值：** [1.5, 0.7, 0.4]

**运行：**
```python
from BVM.main import run_example_2
run_example_2()
```

**挑战：**
- 参数空间复杂
- 观测点较多（50 个）
- 严格的 ε 可能导致后验多峰

---

### 算例 3: Energy Dissipation Model (能量耗散模型)

**问题描述：**

结构在循环荷载下的能量耗散：

```
E(F, m, kₙ, k) = m × F^(log(kₙ)) + k × F²
```

其中：
- **F**: 外加力
- **m, kₙ, k**: 待识别的材料参数

**数据来源：** 混凝土试件的准静态循环试验

**先验分布：**
- m ~ N(1.20, 0.09²)
- log(kₙ) ~ N(5.61, 0.40²)
- k ~ N(1172700, 13760²)

**测试 ε 值：** [0.005, 0.001, 0.0005]

**运行：**
```python
from BVM.main import run_example_3
run_example_3()
```

**工程意义：**
- 用于结构健康监测
- 参数反映材料退化程度
- ε 对应测量传感器精度

---

### 运行所有算例

```bash
cd BayesCalibration/BVM
python main.py
```

所有结果将保存在 `results/` 目录下。

---

## API 参考

### `BVMLikelihood`

```python
BVMLikelihood(
    observed_input: np.ndarray,
    observed_output_mean: np.ndarray,
    observed_output_std: np.ndarray,
    model: callable,
    epsilon: float
)
```

**参数：**
- `observed_input`: 观测输入 X，shape (n_obs, x_dim)
- `observed_output_mean`: 观测输出均值，shape (n_obs,)
- `observed_output_std`: 观测输出噪声标准差，shape (n_obs,)
  - 如果是确定性数据，传入 0 或极小值
- `model`: 计算模型函数
  - 签名: `f(input_concat) -> predictions`
  - `input_concat`: shape (n, x_dim + theta_dim)，每行是 [x, θ]
  - 返回: shape (n,)，预测值
- `epsilon`: 一致性容差

**方法：**

#### `log_likelihood(theta: np.ndarray) -> np.ndarray`
计算对数似然值。

**输入：**
- `theta`: shape (n_samples, n_dim) 或 (n_dim,)

**返回：**
- `log_lik`: shape (n_samples, 1)

**示例：**
```python
bvm_lik = BVMLikelihood(x_obs, y_obs, y_std, model, epsilon=1.0)
log_lik = bvm_lik.log_likelihood(theta_samples)
```

---

#### `evaluate(theta: np.ndarray) -> np.ndarray`
计算原始似然值（非对数）。

**返回：**
- `likelihood`: shape (n_samples, 1)

---

### `BVMValidator`

```python
BVMValidator(
    model_func: callable,
    val_input: np.ndarray,
    val_output_mean: np.ndarray,
    val_output_std: np.ndarray = None
)
```

**参数：**
- `model_func`: 模型函数（同 `BVMLikelihood`）
- `val_input`: 验证集输入
- `val_output_mean`: 验证集输出
- `val_output_std`: 验证集噪声（可选）

**方法：**

#### `compute_bvm_score(theta_samples, epsilon, mode='strict') -> float`

**参数：**
- `theta_samples`: 参数样本（通常是后验样本）
- `epsilon`: 验证容差
- `mode`: 验证模式
  - `'strict'`: 严格一致性（所有点同时满足）
  - `'reliability'`: 平均可靠性（逐点平均）

**返回：**
- `score`: 一致性概率，范围 [0, 1]

**示例：**
```python
validator = BVMValidator(model, x_val, y_val, y_std_val)
strict_score = validator.compute_bvm_score(posterior, epsilon=1.0, mode='strict')
print(f"Strict Agreement: {strict_score:.2%}")
```

---

#### `report(theta_samples, epsilon)`
打印完整的验证报告（包含两种模式）。

**输出：**
```
--- BVM Validation Report (eps=1.0) ---
Strict Agreement Probability (Model Testing): 0.892341
Reliability Metric (Average Pointwise):       0.976543
```

---

### 模型函数接口

所有 BVM 函数都要求模型符合以下接口：

```python
def model_func(input_concat: np.ndarray) -> np.ndarray:
    """
    参数:
    ----------
    input_concat : np.ndarray, shape (n, x_dim + theta_dim)
        每行格式: [x₁, x₂, ..., θ₁, θ₂, ...]

    返回:
    ----------
    predictions : np.ndarray, shape (n,)
        模型预测值
    """
    x = input_concat[:, :x_dim]       # 提取输入变量
    theta = input_concat[:, x_dim:]   # 提取参数

    # 计算预测（向量化）
    y_pred = ...  # your model computation

    return y_pred
```

**示例模型：**

```python
# 一维输入，二维参数 (a, b)
def linear_model(input_concat):
    x = input_concat[:, 0]
    a = input_concat[:, 1]
    b = input_concat[:, 2]
    return a * x + b

# 二维输入，三维参数
def nonlinear_model(input_concat):
    x1 = input_concat[:, 0]
    x2 = input_concat[:, 1]
    theta1 = input_concat[:, 2]
    theta2 = input_concat[:, 3]
    theta3 = input_concat[:, 4]
    return theta1 * np.exp(-theta2 * x1) * np.sin(theta3 * x2)
```

---

## 参考文献

**主要参考文献：**

```
Nagel, J. B., & Sudret, B. (2016)
"A Unified Framework for Multilevel Uncertainty Quantification in Bayesian Inverse Problems"
Probabilistic Engineering Mechanics, 43:68-84
DOI: 10.1016/j.probengmech.2015.09.007
```

**相关文献：**

1. **Bayesian Validation Metric:**
   - Ferson, S., Oberkampf, W. L., & Ginzburg, L. (2008). "Model validation and predictive capability for the thermal challenge problem." Computer Methods in Applied Mechanics and Engineering, 197(29-32):2408-2430.

2. **Transitional MCMC:**
   - Ching, J., & Chen, Y.-C. (2007). "Transitional Markov Chain Monte Carlo Method for Bayesian Model Updating, Model Class Selection, and Model Averaging." Journal of Engineering Mechanics, 133(7):816-832.

3. **贝叶斯系统识别:**
   - Beck, J. L., & Au, S.-K. (2002). "Bayesian updating of structural models and reliability using Markov chain Monte Carlo simulation." Journal of Engineering Mechanics, 128(4):380-391.

4. **广义贝叶斯推断:**
   - Bissiri, P. G., Holmes, C. C., & Walker, S. G. (2016). "A general framework for updating belief distributions." Journal of the Royal Statistical Society: Series B, 78(5):1103-1130.

---

## 常见问题 (FAQ)

### 1. BVM 与传统高斯似然有什么区别？

| 特性 | 高斯似然 | BVM 似然 |
|------|---------|---------|
| **形式** | `L ∝ exp(-0.5×χ²)` | `L = ∏ P(|y-f| ≤ ε)` |
| **参数** | 需要估计 σ | 用户指定 ε |
| **鲁棒性** | 对异常值敏感 | 鲁棒（容差带内一视同仁） |
| **适用场景** | 误差已知为正态 | 模型有偏差或测量精度有限 |
| **物理意义** | 最小化平方误差 | 最大化一致性概率 |

**何时选择 BVM？**
- 模型简化或有系统偏差
- 工程容差比统计误差更重要
- 希望控制"可接受误差范围"而非"最优拟合"

---

### 2. 如何选择合适的 ε？

**方法 1: 基于测量精度**
```python
epsilon = measurement_precision  # 例如：传感器精度 ±0.5mm
```

**方法 2: 基于数据噪声**
```python
epsilon = 2.0 * np.std(y_obs)  # 2 倍标准差
```

**方法 3: 敏感性分析**
```python
epsilon_list = [0.1, 0.05, 0.01]
for eps in epsilon_list:
    # 运行 BVM 校准
    # 观察后验收缩和验证分数
```

**经验法则：**
- ε 太大 → 后验接近先验（欠拟合）
- ε 太小 → 后验过窄或无解（过拟合）
- 合适的 ε → 后验明显收缩但 Reliability > 0.8

---

### 3. BVM 能处理多输出问题吗？

可以！有两种方式：

**方式 1: 分别校准（推荐）**
```python
# 对每个输出变量单独创建 BVM 似然
bvm_lik_1 = BVMLikelihood(x, y1, sigma1, model_1, eps1)
bvm_lik_2 = BVMLikelihood(x, y2, sigma2, model_2, eps2)

# 联合似然 = 两个似然的乘积（对数空间求和）
class JointLikelihood(LikelihoodFunction):
    def log_likelihood(self, theta):
        return bvm_lik_1.log_likelihood(theta) + bvm_lik_2.log_likelihood(theta)

joint_lik = JointLikelihood(lambda x: None)
```

**方式 2: 向量化模型**
```python
# 模型返回多个输出
def multi_output_model(input_concat):
    # 返回 shape (n, n_outputs)
    return np.column_stack([output1, output2, ...])

# 将多输出展平为单输出
y_obs_flat = np.concatenate([y1, y2, ...])
```

---

### 4. 为什么验证分数很低？

**可能原因：**

1. **ε 设置过小**
   - 解决：增大 ε 或检查数据质量

2. **模型不匹配**
   - 解决：改进模型或考虑模型偏差

3. **先验过宽**
   - 解决：使用更informative的先验

4. **数据不足**
   - 解决：增加观测点或使用正则化

**诊断步骤：**
```python
# 1. 检查先验预测
prior_samples = prior.sample(1000)
validator.report(prior_samples, epsilon)  # 先验分数

# 2. 检查后验预测
validator.report(posterior_samples, epsilon)  # 后验分数

# 3. 如果后验分数仍很低，尝试增大 ε
validator.report(posterior_samples, epsilon * 2)
```

---

### 5. BVM 可以用于模型选择吗？

可以！使用 **Strict Agreement Probability**：

```python
# 模型 A
bvm_lik_A = BVMLikelihood(..., model=model_A, epsilon=eps)
posterior_A = tmcmc.sample(bvm_lik_A, prior_A, ...)
score_A = validator.compute_bvm_score(posterior_A, eps, mode='strict')

# 模型 B
bvm_lik_B = BVMLikelihood(..., model=model_B, epsilon=eps)
posterior_B = tmcmc.sample(bvm_lik_B, prior_B, ...)
score_B = validator.compute_bvm_score(posterior_B, eps, mode='strict')

# 比较
print(f"Model A: {score_A:.4f}")
print(f"Model B: {score_B:.4f}")
print(f"Preferred: {'A' if score_A > score_B else 'B'}")
```

**注意：**
- 使用相同的 ε 和验证集
- Strict 分数对模型形式敏感
- 考虑结合其他指标（如 AIC/BIC）

---
