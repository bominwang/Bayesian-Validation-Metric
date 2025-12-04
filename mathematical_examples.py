import numpy as np
from scipy.optimize import brentq
# ======================================================================================================================
# 算例 1: Bacterial Growth Model (Monod Model)
# ======================================================================================================================

class BacterialGrowthModel:
    """
    算例 1: 细菌生长模型 (Monod Model)
    Equation (19): y = (alpha1 * x) / (alpha2 + x)
    """

    def __init__(self):
        # 观测数据
        self.x_data = np.array([28, 55, 83, 110, 138, 225, 375]).reshape(-1, 1)
        self.y_data = np.array([0.053, 0.060, 0.112, 0.105, 0.099, 0.122, 0.125])
        self.y_std_zeros = np.zeros_like(self.y_data)

    @staticmethod
    def model(inputs: np.ndarray):
        """
        Monod 模型计算函数
        参数:
        inputs: np.ndarray, shape (N, 3)
            - column 0: x (concentration)
            - column 1: alpha1 (maximum growth rate)
            - column 2: alpha2 (saturation constant)
        返回:
        y_pred: np.ndarray, shape (N,)
        """
        # 解包输入
        x = inputs[:, 0]
        alpha1 = inputs[:, 1]
        alpha2 = inputs[:, 2]
        y_pred = (alpha1 * x) / (alpha2 + x + 1e-12)
        return y_pred

# ======================================================================================================================
# 算例 2: Synthetic Toy Model
# ======================================================================================================================

class ToyModel:
    """
    算例 2: 合成玩具模型
    Equation (22): y = a1 + a2*x*exp(-a3*cos(a4*x)) + a5*sin(a6*x)
    """

    def __init__(self):
        pass

    @staticmethod
    def generate_data(n_points=50):
        """
        y(x) = 1 + x*exp(-cos(10x)) + sin(10x) + epsilon
        """
        np.random.seed(42)
        x = np.sort(np.random.uniform(0, 3, n_points)).reshape(-1, 1)
        y_true = 1 + x.flatten() * np.exp(-np.cos(10 * x.flatten())) + np.sin(10 * x.flatten())

        # 添加 Aleatoric 不确定性
        # x in [0, 1.5]: N(0, 0.4^2)
        # x in [1.5, 3]: N(0, 0.6^2)
        std_aleatoric = np.where(x.flatten() <= 1.5, 0.4, 0.6)
        noise_aleatoric = np.random.normal(0, std_aleatoric)
        # 添加 Epistemic 测量不确定性 N(0, 0.5^2)
        noise_epistemic = np.random.normal(0, 0.5, size=n_points)
        # 观测值 y
        y_obs = y_true + noise_aleatoric + noise_epistemic
        # 总方差 = Aleatoric^2 + Epistemic^2
        total_std = np.sqrt(std_aleatoric ** 2 + 0.5 ** 2)
        return x, y_obs, total_std

    @staticmethod
    def model(inputs):
        """
        Toy 模型计算函数 (Eq 22)

        参数:
        inputs: np.ndarray, shape (N, 7)
            - column 0: x
            - column 1-6: alpha1, alpha2, alpha3, alpha4, alpha5, alpha6
        """
        x = inputs[:, 0]
        a1 = inputs[:, 1]
        a2 = inputs[:, 2]
        a3 = inputs[:, 3]
        a4 = inputs[:, 4]
        a5 = inputs[:, 5]
        a6 = inputs[:, 6]
        term_exp = np.exp(-a3 * np.cos(a4 * x))
        term_sin = np.sin(a6 * x)
        y_pred = a1 + a2 * x * term_exp + a5 * term_sin
        return y_pred


# ======================================================================================================================
# 算例 3: Energy Dissipation Model (Small-wood Model)
# 对应论文 Section 4.2.3
# ======================================================================================================================

class EnergyDissipationModel:
    """
    算例 3: 能量耗散模型 (Small-wood Model)
    目标: 预测每个循环的能量耗散 D_E

    参数:
    - m: 非线性指数 (无单位)
    - kn: 非线性刚度 (lbf/in^m) -> 为了优化方便，通常输入 log10(kn)
    - k: 线性刚度 (lbf/in)

    输入: F (Force)
    """

    def __init__(self):
        # 论文 Table 4 中的校准数据
        self.F_data = np.array([60, 120, 180, 240, 320]).reshape(-1, 1)  # Force (lbf)
        self.E_data = np.array([5.30e-5, 2.85e-4, 7.78e-4, 1.55e-3, 2.50e-3])  # Energy (lbf*in)
        self.E_std_zeros = np.zeros_like(self.E_data)

    @staticmethod
    def _solve_delta_z(F, k, kn, m):
        """
        求解方程 (24): 2F = k * dz - kn * dz^m
        求 dz (位移幅值)
        注意: 这里需要数值求解根。
        """

        # 定义目标函数: f(dz) = k*dz - kn*dz^m - 2F = 0
        def eq(dz):
            return k * dz - kn * (dz ** m) - 2 * F

        # 寻找根。物理上 dz 必须为正。
        # 上界估计: 当 kn=0时, dz = 2F/k。由于 kn*dz^m 通常减小刚度(软化)或增加(硬化)，
        # 这里需要一个鲁棒的区间。
        # 假设是一个典型的机械连接，位移一般很小。
        try:
            # 尝试在 [1e-9, 10.0] 范围内寻找根 (inch)
            root = brentq(eq, 1e-9, 10.0)
            return root
        except ValueError:
            return np.nan  # 求解失败

    @staticmethod
    def model(inputs):
        """
        能量耗散模型计算函数

        参数:
        inputs: np.ndarray, shape (N, 4)
            - column 0: F (Force Input)
            - column 1: m
            - column 2: log10_kn (论文中使用 log10(kn) 作为参数)
            - column 3: k

        注意:
        1. 由于需要数值求解方程根，此函数包含循环，速度较慢。
        2. 为了兼容 BVM 的向量化调用，我们在内部进行循环处理。
        """
        # 获取输入维度
        n_samples = inputs.shape[0]
        y_pred = np.zeros(n_samples)
        F_vec = inputs[:, 0]
        m_vec = inputs[:, 1]
        log_kn_vec = inputs[:, 2]
        k_vec = inputs[:, 3]
        # 还原 kn
        kn_vec = 10 ** log_kn_vec
        # 遍历计算
        for i in range(n_samples):
            F = F_vec[i]
            m = m_vec[i]
            kn = kn_vec[i]
            k = k_vec[i]

            # 1. 求解 Delta z (Eq 24)
            # 2F = k*dz - kn*dz^m
            try:
                est = 2 * F / k
                # 搜索范围设定在估计值的附近
                dz = brentq(lambda z: k * z - kn * (z ** m) - 2 * F, 1e-10, est * 20.0)
            except (ValueError, RuntimeError):
                # 如果求解失败（参数不物理），返回极大误差或NaN
                dz = np.nan

            if np.isnan(dz):
                y_pred[i] = 1e9
                continue
            # 2. 计算 Energy Dissipation
            DE = kn * ((m - 1) / (m + 1)) * (dz ** (m + 1))
            y_pred[i] = DE

        return y_pred