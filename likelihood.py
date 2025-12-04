import numpy as np
from scipy import stats

class LikelihoodFunction:
    """
    似然函数类
    用于封装用户定义的似然函数
    """

    def __init__(self, func: callable):
        """
        初始化似然函数
        """
        self.func = func

    def evaluate(self, theta: np.ndarray) -> np.ndarray:
        """计算似然值"""
        return self.func(theta)

    def log_likelihood(self, theta: np.ndarray) -> np.ndarray:
        """计算对数似然值"""
        likelihood = self.func(theta)
        # 避免 log(0)
        return np.log(likelihood + 1e-300)

    def __repr__(self):
        return f"LikelihoodFunction(func={self.func.__name__})"


class BVMLikelihood(LikelihoodFunction):
    """
    贝叶斯验证度量 (BVM) 似然函数
    继承自 LikelihoodFunction，重写了 log_likelihood 计算逻辑。
    """

    def __init__(self,
                 observed_input: np.ndarray,
                 observed_output_mean: np.ndarray,
                 observed_output_std: np.ndarray,
                 model: callable,
                 epsilon: float):
        """
        初始化 BVM 似然函数

        参数:
        ----------
        observed_input : np.ndarray, shape (n_obs, x_dim)
        observed_output_mean : np.ndarray, shape (n_obs,)
        observed_output_std : np.ndarray, shape (n_obs,)
        model : callable
        epsilon : float
        """
        self.observed_input = observed_input
        self.observed_output_mean = observed_output_mean

        # 数值稳定性处理：防止 sigma 为 0 导致除以 0 错误
        self.observed_output_std = np.where(observed_output_std < 1e-9, 1e-9, observed_output_std)

        self.model_func = model
        self.epsilon = epsilon
        self.n_obs = len(observed_output_mean)

        # 调用父类构造函数
        super().__init__(lambda theta: self.evaluate(theta))

    def _predict_batch(self, theta: np.ndarray) -> np.ndarray:
        """
        执行批量预测
        """
        n_particles = theta.shape[0]
        n_obs = self.n_obs
        # 1. 处理 Theta: 按行重复
        # shape: (n_particles * n_obs, theta_dim)
        theta_repeated = np.repeat(theta, n_obs, axis=0)
        # 2. 处理 X: 整体堆叠
        # shape: (n_particles * n_obs, x_dim)
        x_tiled = np.tile(self.observed_input, (n_particles, 1))
        # 3. 拼接输入
        # shape: (n_particles * n_obs, x_dim + theta_dim)
        input_concat = np.hstack([x_tiled, theta_repeated])
        # 4. 调用模型
        y_pred_flat = self.model_func(input_concat)
        # 5. 还原形状 -> (n_particles, n_obs)
        y_pred = y_pred_flat.reshape(n_particles, n_obs)

        return y_pred

    def log_likelihood(self, theta: np.ndarray) -> np.ndarray:
        """
        计算对数似然
        """
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)

        # 1. 获取模型预测
        y_pred = self._predict_batch(theta)

        # 2. 准备数据
        mu = self.observed_output_mean
        sigma = self.observed_output_std
        eps = self.epsilon

        # 3. 计算 Z-score
        z_upper = (y_pred - mu + eps) / sigma
        z_lower = (y_pred - mu - eps) / sigma

        # 4. 计算区间概率 (Probability of Agreement)
        prob_matrix = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)

        # 5. 数值稳定性处理
        prob_matrix = np.maximum(prob_matrix, 1e-300)

        # 6. 计算对数似然 (各观测点对数概率之和)
        log_lik = np.sum(np.log(prob_matrix), axis=1).reshape(-1, 1)
        return log_lik

    def evaluate(self, theta: np.ndarray) -> np.ndarray:
        """
        计算原始似然值
        """
        return np.exp(self.log_likelihood(theta))

    def __repr__(self):
        return (f"BVMLikelihood(n_obs={self.n_obs}, epsilon={self.epsilon}, "
                f"type={'Noisy' if np.any(self.observed_output_std > 1e-8) else 'Deterministic'})")