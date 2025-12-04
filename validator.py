import numpy as np
from scipy import stats


class BVMValidator:
    """
    BVM 模型验证工具
    功能:
    基于给定的参数样本(Posterior/Prior)，计算模型预测与试验数据的一致性概率。
    """

    def __init__(self,
                 model_func: callable,
                 val_input: np.ndarray,
                 val_output_mean: np.ndarray,
                 val_output_std: np.ndarray = None):
        """
        初始化验证器

        参数:
        ----------
        model_func : callable
            模型函数，接受拼接后的 [x, theta] 输入
        val_input : np.ndarray
            验证集输入 X
        val_output_mean : np.ndarray
            验证集输出观测值 Y
        val_output_std : np.ndarray, optional
            验证集输出噪声 Sigma (如果是确定性数据，可不传或传0)
        """
        self.model_func = model_func
        self.val_input = val_input
        self.val_output_mean = val_output_mean

        # 处理数据噪声：如果是 None 或 0，替换为极小值以防除零
        if val_output_std is None:
            self.val_output_std = np.full_like(val_output_mean, 1e-9)
        else:
            self.val_output_std = np.where(val_output_std < 1e-9, 1e-9, val_output_std)

        self.n_obs = len(val_output_mean)

    def _predict_batch(self, theta_samples: np.ndarray) -> np.ndarray:
        """
        批量运行模型预测
        """
        n_samples = theta_samples.shape[0]
        n_obs = self.n_obs

        # 1. 构造批量输入 (n_samples * n_obs, dim)
        # 将参数重复 n_obs 次
        theta_repeated = np.repeat(theta_samples, n_obs, axis=0)
        # 将输入 X 堆叠 n_samples 次
        x_tiled = np.tile(self.val_input, (n_samples, 1))

        input_concat = np.hstack([x_tiled, theta_repeated])

        # 2. 运行模型
        y_pred_flat = self.model_func(input_concat)

        # 3. 重塑为 [n_samples, n_obs]
        y_preds = y_pred_flat.reshape(n_samples, n_obs)

        return y_preds

    def compute_bvm_score(self, theta_samples: np.ndarray, epsilon: float, mode: str = 'strict') -> float:
        """
        计算 BVM 验证分数 (一致性概率)

        参数:
        ----------
        theta_samples : np.ndarray
            参数样本 (例如 TMCMC 的后验样本，或先验采样)
        epsilon : float
            用户定义的一致性容差
        mode : str
            - 'strict': 要求一条模型曲线必须同时通过所有数据点的容差带才算一致。
                        这通常用于模型选择，分数较低但区分度高。
            - 'reliability': 计算平均每个点的一致性概率。
                             回答“平均而言，模型有多大概率能预测对一个点”。

        返回:
        ----------
        score : float (0.0 ~ 1.0)
        """
        # 1. 获取预测结果 [n_samples, n_obs]
        y_preds = self._predict_batch(theta_samples)

        # 2. 准备广播变量
        mu = self.val_output_mean
        sigma = self.val_output_std

        # 3. 计算每个点的一致性概率 P_i
        # 利用 CDF 计算区间概率: P(|y_pred - y_data| <= epsilon)
        z_upper = (y_preds - mu + epsilon) / sigma
        z_lower = (y_preds - mu - epsilon) / sigma

        # prob_matrix 形状: [n_samples, n_obs]
        prob_matrix = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)

        # 4. 根据模式聚合概率
        if mode == 'strict':
            # 逻辑：样本 j "一致" 当且仅当它对所有观测点 i 都一致
            # P(sample_j agrees) = P_j,1 * P_j,2 * ... * P_j,n (假设观测点独立)
            sample_agreement_prob = np.prod(prob_matrix, axis=1)
            # 最终分数是对所有参数样本求期望
            bvm_score = np.mean(sample_agreement_prob)

        elif mode == 'reliability':
            # 逻辑：不要求全部匹配，只关心平均匹配程度
            # 直接对矩阵所有元素求平均
            bvm_score = np.mean(prob_matrix)

        else:
            raise ValueError("Mode must be 'strict' or 'reliability'")

        return bvm_score

    def report(self, theta_samples: np.ndarray, epsilon: float):
        """
        打印简单的验证报告
        """
        strict_score = self.compute_bvm_score(theta_samples, epsilon, mode='strict')
        reli_score = self.compute_bvm_score(theta_samples, epsilon, mode='reliability')
        print(f"--- BVM Validation Report (eps={epsilon}) ---")
        print(f"Strict Agreement Probability (Model Testing): {strict_score:.6f}")
        print(f"Reliability Metric (Average Pointwise):       {reli_score:.6f}")
        return strict_score, reli_score