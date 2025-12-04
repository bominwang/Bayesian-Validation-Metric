"""
------------------------------------------------------------------------------------------------------------------------
BVM-TMCMC 参数修正与验证主程序 (Calibration & Validation)
------------------------------------------------------------------------------------------------------------------------
复现 Paper: "A Generalized Bayesian Approach to Model Calibration"

流程:
1. Calibration (修正): 使用 TMCMC + BVMLikelihood 获取后验样本。
2. Validation (验证): 使用 BVMValidator 计算一致性概率 (Strict & Reliability)。
3. Visualization (绘图): 绘制预测包络和参数后验。

Outputs: 结果图像保存在 ./results/ 目录下
------------------------------------------------------------------------------------------------------------------------
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ======================================================================================================================
# 导入模块
# ======================================================================================================================
from BayesCalibration.TMCMC.utils import PriorDistribution, JointPrior
from BayesCalibration.TMCMC.tmcmc import TransitionalMCMC
from BayesCalibration.BVM.likelihood import BVMLikelihood
from BayesCalibration.BVM.validator import BVMValidator
from BayesCalibration.BVM.mathematical_examples import BacterialGrowthModel, ToyModel, EnergyDissipationModel

# 设置绘图风格
plt.style.use('seaborn-v0_8-paper')
SAVE_DIR = 'results'
os.makedirs(SAVE_DIR, exist_ok=True)


# ======================================================================================================================
# 通用计算与绘图函数
# ======================================================================================================================

def evaluate_model_on_grid(model_func, samples, x_grid):
    """
    辅助函数：在给定网格 x_grid 上评估一批参数 samples 的模型输出
    """
    if x_grid.ndim == 1:
        x_grid = x_grid.reshape(-1, 1)

    n_grid = x_grid.shape[0]
    n_samples = samples.shape[0]

    # 限制用于绘图的样本数量以提高速度
    max_plot_samples = 500
    if n_samples > max_plot_samples:
        indices = np.random.choice(n_samples, max_plot_samples, replace=False)
        samples_to_run = samples[indices]
        n_samples_run = max_plot_samples
    else:
        samples_to_run = samples
        n_samples_run = n_samples

    # 构造输入矩阵
    x_tiled = np.tile(x_grid, (n_samples_run, 1))
    theta_repeated = np.repeat(samples_to_run, n_grid, axis=0)
    input_concat = np.hstack([x_tiled, theta_repeated])

    # 运行模型
    try:
        y_flat = model_func(input_concat)
        y_preds = y_flat.reshape(n_samples_run, n_grid)
    except Exception as e:
        print(f"    ! Vectorized eval failed ({e}), fallback to loop...")
        y_preds = np.zeros((n_samples_run, n_grid))
        for i in range(n_samples_run):
            theta_block = np.tile(samples_to_run[i], (n_grid, 1))
            inp = np.hstack([x_grid, theta_block])
            y_preds[i, :] = model_func(inp)

    return y_preds


def plot_envelope(samples, model_func, x_grid, x_obs, y_obs, y_std,
                  color_theme, title, filename, xlabel="x", ylabel="y"):
    """
    绘制单一状态（先验或后验）的预测包络图
    """
    print(f"  -> Plotting: {filename}")

    # 1. 计算预测
    y_preds = evaluate_model_on_grid(model_func, samples, x_grid)

    # 2. 计算统计量
    mu_pred = np.mean(y_preds, axis=0)
    p025 = np.percentile(y_preds, 2.5, axis=0)
    p975 = np.percentile(y_preds, 97.5, axis=0)

    # 3. 设置颜色
    if color_theme == 'red':
        fill_color, line_color, fill_alpha = '#e74c3c', '#c0392b', 0.2
        label_ci, label_mean = 'Prior 95% CI', 'Prior Mean'
    else:
        fill_color, line_color, fill_alpha = '#3498db', '#2980b9', 0.5
        label_ci, label_mean = 'Posterior 95% CI', 'Posterior Mean'

    # 4. 绘图
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(x_grid.flatten(), p025, p975, color=fill_color, alpha=fill_alpha, label=label_ci)
    ax.plot(x_grid.flatten(), mu_pred, color=line_color, linestyle='--', linewidth=2, label=label_mean)

    # 绘制数据
    if y_std is not None and np.any(y_std > 1e-6):
        if y_std.ndim == 1:
            yerr = y_std
        else:
            yerr = y_std.flatten()
        ax.errorbar(x_obs.flatten(), y_obs.flatten(), yerr=yerr, fmt='ko',
                    markersize=4, capsize=3, label='Data', zorder=10)
    else:
        ax.plot(x_obs.flatten(), y_obs.flatten(), 'ko', markersize=5, label='Data', zorder=10)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=300)
    plt.close()


def plot_parameter_posterior(posterior_samples, param_names, title_prefix, filename_prefix):
    """
    绘制参数后验直方图
    """
    n_dim = posterior_samples.shape[1]
    n_plot = min(n_dim, 6)

    fig, axes = plt.subplots(1, n_plot, figsize=(4 * n_plot, 4))
    if n_plot == 1: axes = [axes]

    for i in range(n_plot):
        ax = axes[i]
        ax.hist(posterior_samples[:, i], bins=30, density=True, color='#3498db', alpha=0.6, edgecolor='white')
        try:
            if np.std(posterior_samples[:, i]) > 1e-9:
                kde = stats.gaussian_kde(posterior_samples[:, i])
                x_range = np.linspace(posterior_samples[:, i].min(), posterior_samples[:, i].max(), 100)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2)
        except:
            pass
        ax.set_title(f'{param_names[i]}')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{title_prefix} Parameters", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{filename_prefix}_params.png"), dpi=300)
    plt.close()


# ======================================================================================================================
# 算例 1: Bacterial Growth Model
# ======================================================================================================================
def run_example_1():
    print("\n" + "#" * 100)
    print("Example 1: Bacterial Growth Model")
    print("#" * 100)

    # 1. 初始化
    bg_model = BacterialGrowthModel()
    prior = JointPrior([
        PriorDistribution('normal', mean=0.17, std=0.025),  # alpha1
        PriorDistribution('normal', mean=47.5, std=3.0)  # alpha2
    ])

    # 2. 绘制先验
    print("Sampling Prior...")
    prior_samples = prior.sample(1000)
    x_grid = np.linspace(0, 400, 100)
    plot_envelope(
        prior_samples, bg_model.model, x_grid, bg_model.x_data, bg_model.y_data, None,
        color_theme='red', title="Ex1: Prior Prediction", filename="ex1_prior.png",
        xlabel="Substrate Concentration", ylabel="Growth Rate"
    )

    # 3. 循环测试 Epsilon
    epsilon_list = [0.1, 0.03, 0.01]

    # 初始化验证器
    validator = BVMValidator(
        model_func=bg_model.model,
        val_input=bg_model.x_data,
        val_output_mean=bg_model.y_data,
        val_output_std=bg_model.y_std_zeros
    )

    for eps in epsilon_list:
        print(f"\n---> Running with epsilon = {eps} ...")

        # 3.1 修正 (Calibration)
        bvm_lik = BVMLikelihood(
            observed_input=bg_model.x_data,
            observed_output_mean=bg_model.y_data,
            observed_output_std=bg_model.y_std_zeros,
            model=bg_model.model,
            epsilon=eps
        )
        init_samples = prior.sample(2000)
        tmcmc = TransitionalMCMC(initial_beta=0.2, target_cov=1.0, verbose=False)
        posterior = tmcmc.sample(bvm_lik, prior, init_samples, n_mh_steps=20)

        # 3.2 验证 (Validation)
        print(f"  -> Validating Model (Posterior)...")
        # 使用相同的 epsilon 进行验证（也可设定为不同的验证标准）
        validator.report(posterior, eps)

        # 3.3 绘图
        plot_envelope(
            posterior, bg_model.model, x_grid, bg_model.x_data, bg_model.y_data, None,
            color_theme='blue', title=f"Ex1: Posterior ($\epsilon={eps}$)",
            filename=f"ex1_eps_{eps}_posterior.png",
            xlabel="Substrate Concentration", ylabel="Growth Rate"
        )
        plot_parameter_posterior(posterior, [r'$\alpha_1$', r'$\alpha_2$'],
                                 f"Ex1 (eps={eps})", f"ex1_eps_{eps}")


# ======================================================================================================================
# 算例 2: Toy Model
# ======================================================================================================================
def run_example_2():
    print("\n" + "#" * 100)
    print("Example 2: Synthetic Toy Model")
    print("#" * 100)

    # 1. 初始化
    toy_model = ToyModel()
    x_obs, y_obs, y_std = ToyModel.generate_data(n_points=50)
    means = [1.0, 1.0, 1.0, 10.0, 1.0, 10.0]
    stds = [0.35, 0.3, 0.3, 0.3, 0.3, 0.3]
    prior = JointPrior([PriorDistribution('normal', mean=m, std=s) for m, s in zip(means, stds)])

    # 2. 绘制先验
    print("Sampling Prior...")
    prior_samples = prior.sample(1000)
    x_grid = np.linspace(0, 3, 100)
    plot_envelope(
        prior_samples, ToyModel.model, x_grid, x_obs, y_obs, y_std,
        color_theme='red', title="Ex2: Prior Prediction", filename="ex2_prior.png",
        xlabel="x", ylabel="y"
    )

    # 3. 循环测试
    epsilon_list = [1.5, 0.7, 0.4]

    # 初始化验证器 (注意传入 y_std)
    validator = BVMValidator(
        model_func=ToyModel.model,
        val_input=x_obs,
        val_output_mean=y_obs,
        val_output_std=y_std
    )

    for eps in epsilon_list:
        print(f"\n---> Running with epsilon = {eps} ...")

        # 3.1 修正
        bvm_lik = BVMLikelihood(
            observed_input=x_obs,
            observed_output_mean=y_obs,
            observed_output_std=y_std,
            model=ToyModel.model,
            epsilon=eps
        )
        init_samples = prior.sample(2000)
        tmcmc = TransitionalMCMC(initial_beta=0.2, target_cov=1.0, verbose=False)
        posterior = tmcmc.sample(bvm_lik, prior, init_samples, n_mh_steps=20)

        # 3.2 验证
        print(f"  -> Validating Model (Posterior)...")
        validator.report(posterior, eps)

        # 3.3 绘图
        plot_envelope(
            posterior, ToyModel.model, x_grid, x_obs, y_obs, y_std,
            color_theme='blue', title=f"Ex2: Posterior ($\epsilon={eps}$)",
            filename=f"ex2_eps_{eps}_posterior.png",
            xlabel="x", ylabel="y"
        )
        plot_parameter_posterior(posterior[:, :3], ['a1', 'a2', 'a3'],
                                 f"Ex2 (eps={eps})", f"ex2_eps_{eps}")


# ======================================================================================================================
# 算例 3: Energy Dissipation Model
# ======================================================================================================================
def run_example_3():
    print("\n" + "#" * 100)
    print("Example 3: Energy Dissipation Model")
    print("#" * 100)

    # 1. 初始化
    ed_model = EnergyDissipationModel()
    prior = JointPrior([
        PriorDistribution('normal', mean=1.20, std=0.09),
        PriorDistribution('normal', mean=5.61, std=0.40),
        PriorDistribution('normal', mean=1172700, std=13760)
    ])

    # 2. 绘制先验
    print("Sampling Prior...")
    prior_samples = prior.sample(1000)
    x_grid = np.linspace(50, 330, 50)
    plot_envelope(
        prior_samples, ed_model.model, x_grid, ed_model.F_data, ed_model.E_data, None,
        color_theme='red', title="Ex3: Prior Prediction", filename="ex3_prior.png",
        xlabel="Force F", ylabel="Energy"
    )

    # 3. 循环测试
    epsilon_list = [0.005, 0.001, 0.0005]

    # 初始化验证器
    validator = BVMValidator(
        model_func=ed_model.model,
        val_input=ed_model.F_data,
        val_output_mean=ed_model.E_data,
        val_output_std=ed_model.E_std_zeros
    )

    for eps in epsilon_list:
        print(f"\n---> Running with epsilon = {eps} ...")

        # 3.1 修正
        bvm_lik = BVMLikelihood(
            observed_input=ed_model.F_data,
            observed_output_mean=ed_model.E_data,
            observed_output_std=ed_model.E_std_zeros,
            model=ed_model.model,
            epsilon=eps
        )
        init_samples = prior.sample(2000)
        tmcmc = TransitionalMCMC(initial_beta=0.2, target_cov=1.0, verbose=False)
        posterior = tmcmc.sample(bvm_lik, prior, init_samples, n_mh_steps=10)

        # 3.2 验证
        print(f"  -> Validating Model (Posterior)...")
        validator.report(posterior, eps)

        # 3.3 绘图
        plot_envelope(
            posterior, ed_model.model, x_grid, ed_model.F_data, ed_model.E_data, None,
            color_theme='blue', title=f"Ex3: Posterior ($\epsilon={eps}$)",
            filename=f"ex3_eps_{eps}_posterior.png",
            xlabel="Force F", ylabel="Energy"
        )
        plot_parameter_posterior(posterior, ['m', 'log(kn)', 'k'],
                                 f"Ex3 (eps={eps})", f"ex3_eps_{eps}")


# ======================================================================================================================
# Main Execution
# ======================================================================================================================
if __name__ == "__main__":
    np.random.seed(42)

    print("Starting BVM-TMCMC Benchmark Study...")
    print(f"Results will be saved to: {os.path.abspath(SAVE_DIR)}")

    try:
        run_example_1()
    except Exception as e:
        print(f"Error in Ex1: {e}")
        import traceback

        traceback.print_exc()

    try:
        run_example_2()
    except Exception as e:
        print(f"Error in Ex2: {e}")
        import traceback

        traceback.print_exc()

    try:
        run_example_3()
    except Exception as e:
        print(f"Error in Ex3: {e}")
        import traceback

        traceback.print_exc()

    print("\nAll tasks finished.")