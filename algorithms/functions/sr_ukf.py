"""
Robust Square-Root UKF (SR-UKF) with Student-t Measurement Update
鲁棒平方根无迹卡尔曼滤波SOC估计算法

创新点：
    1. SR-UKF: 直接传播协方差平方根，数值稳定性更好
    2. NIS门控: 基于归一化新息平方检测异常值
    3. Student-t鲁棒更新: 重尾分布假设，抗异常值能力强
    4. 自适应自由度: 根据NIS动态调节Student-t的ν参数

State Vector: x = [SOC, U1, U2]^T
    - SOC: State of Charge (0-1)
    - U1: First RC circuit polarization voltage (V)
    - U2: Second RC circuit polarization voltage (V)

Reference:
    - Van der Merwe, R. (2004). Sigma-Point Kalman Filters for Probabilistic Inference
    - Huang, Y. et al. (2017). A Novel Robust Student's t-Based Kalman Filter
    - Plett, G. L. (2006). Sigma-point Kalman filtering for battery management systems

Author: Generated for CALCE dataset
Date: 2024
"""

import numpy as np
from scipy import linalg
from pathlib import Path
from typing import Dict, Optional, Tuple
from .battery_model import BatteryModel2RC
from .ffrls import FFRLS


class RobustSRUKF:
    """
    鲁棒平方根无迹卡尔曼滤波器

    结合SR-UKF的数值稳定性和Student-t分布的鲁棒性，
    并通过NIS门控机制检测和处理异常测量值。
    """

    def __init__(
        self,
        initial_soc: float = 0.8,
        capacity_Ah: float = 2.0,
        sample_time: float = 1.0,
        process_noise: np.ndarray = None,
        measurement_noise: float = 2e-4,
        initial_covariance: np.ndarray = None,
        use_online_param_id: bool = True,
        # UKF参数
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        # 鲁棒参数
        enable_nis_gate: bool = True,
        nis_threshold: float = 6.63,  # χ²(1, 0.99)
        nis_R_scale: float = 10.0,    # NIS超阈值时R放大倍数
        enable_student_t: bool = True,
        nu_init: float = 5.0,         # Student-t初始自由度
        nu_min: float = 2.1,          # 最小自由度（>2保证方差有限）
        nu_max: float = 100.0,        # 最大自由度（接近高斯）
        adaptive_nu: bool = True      # 是否自适应调节ν
    ):
        """
        初始化鲁棒SR-UKF估计器

        Args:
            initial_soc: 初始SOC (0-1)
            capacity_Ah: 电池容量 (Ah)
            sample_time: 采样时间 (s)
            process_noise: 过程噪声协方差矩阵Q (3x3)
            measurement_noise: 测量噪声协方差R (标量)
            initial_covariance: 初始状态协方差矩阵P (3x3)
            use_online_param_id: 是否使用在线参数辨识
            alpha: Sigma点分布参数 (1e-4 ~ 1)
            beta: 分布类型参数，高斯分布时beta=2最优
            kappa: 次要缩放参数
            enable_nis_gate: 是否启用NIS门控
            nis_threshold: NIS门控阈值
            nis_R_scale: NIS超阈值时R的放大倍数
            enable_student_t: 是否启用Student-t鲁棒更新
            nu_init: Student-t初始自由度
            nu_min: 最小自由度
            nu_max: 最大自由度
            adaptive_nu: 是否自适应调节自由度
        """
        # 状态维度
        self.n = 3

        # 初始状态 [SOC, U1, U2]
        self.x = np.array([initial_soc, 0.0, 0.0])

        # 采样时间
        self.dt = sample_time

        # UKF参数
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # 计算lambda和权重
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n
        self._compute_weights()

        # 过程噪声协方差矩阵 Q
        if process_noise is None:
            self.Q = np.array([
                [1e-6, 0, 0],
                [0, 1e-8, 0],
                [0, 0, 1e-8]
            ])
        else:
            self.Q = process_noise.copy()
        self.Q_init = self.Q.copy()

        # 过程噪声平方根
        self.sqrt_Q = np.linalg.cholesky(self.Q)

        # 测量噪声协方差 R
        self.R = measurement_noise
        self.R_init = measurement_noise
        self.sqrt_R = np.sqrt(measurement_noise)

        # 状态协方差平方根 S (P = S @ S.T)
        if initial_covariance is None:
            P = np.array([
                [1e-4, 0, 0],
                [0, 1e-6, 0],
                [0, 0, 1e-6]
            ])
        else:
            P = initial_covariance.copy()
        self.S = np.linalg.cholesky(P)

        # 鲁棒参数
        self.enable_nis_gate = enable_nis_gate
        self.nis_threshold = nis_threshold
        self.nis_R_scale = nis_R_scale
        self.enable_student_t = enable_student_t
        self.nu = nu_init
        self.nu_init = nu_init
        self.nu_min = nu_min
        self.nu_max = nu_max
        self.adaptive_nu = adaptive_nu

        # 电池模型
        self.model = BatteryModel2RC(
            capacity_Ah=capacity_Ah,
            sample_time=sample_time
        )

        # 在线参数辨识器
        self.use_online_param_id = use_online_param_id
        if use_online_param_id:
            self.ffrls = FFRLS(sample_time=sample_time)
        else:
            self.ffrls = None

        # 步数计数器
        self.step = 0

        # 历史记录
        self.history = {
            'SOC': [],
            'U1': [],
            'U2': [],
            'Ut_pred': [],
            'error': [],
            'P': [],
            'R0': [],
            'R1': [],
            'R2': [],
            'NIS': [],
            'nu': [],
            'robust_weight': []
        }

    def _compute_weights(self):
        """计算Sigma点权重"""
        n = self.n
        lambda_ = self.lambda_

        # 均值权重
        self.Wm = np.zeros(2 * n + 1)
        self.Wm[0] = lambda_ / (n + lambda_)
        self.Wm[1:] = 1 / (2 * (n + lambda_))

        # 协方差权重
        self.Wc = np.zeros(2 * n + 1)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        self.Wc[1:] = 1 / (2 * (n + lambda_))

    def _generate_sigma_points(self, x: np.ndarray, S: np.ndarray) -> np.ndarray:
        """
        生成Sigma点（平方根形式）

        Args:
            x: 状态均值 (n,)
            S: 状态协方差平方根 (n, n)，下三角矩阵

        Returns:
            sigma_points: Sigma点矩阵 (n, 2n+1)
        """
        n = self.n
        gamma = np.sqrt(n + self.lambda_)

        sigma_points = np.zeros((n, 2 * n + 1))
        sigma_points[:, 0] = x

        for i in range(n):
            sigma_points[:, i + 1] = x + gamma * S[:, i]
            sigma_points[:, n + i + 1] = x - gamma * S[:, i]

        return sigma_points

    def _qr_update(self, A: np.ndarray) -> np.ndarray:
        """
        QR分解更新，提取下三角矩阵

        Args:
            A: 输入矩阵

        Returns:
            R: 下三角Cholesky因子
        """
        # QR分解
        _, R = np.linalg.qr(A.T)
        # 返回下三角矩阵（R的转置的下三角部分）
        return np.tril(R.T)

    def _cholupdate(self, S: np.ndarray, v: np.ndarray, sign: float = 1.0) -> np.ndarray:
        """
        Cholesky因子的秩1更新（数值稳定版本）

        S_new @ S_new.T = S @ S.T + sign * v @ v.T

        Args:
            S: 当前Cholesky因子 (下三角)
            v: 更新向量
            sign: +1 for update, -1 for downdate

        Returns:
            S_new: 更新后的Cholesky因子
        """
        n = len(v)
        S_new = S.copy()
        v = v.copy()
        eps = 1e-12

        try:
            if sign > 0:
                # Cholesky update (rank-1 update)
                for k in range(n):
                    S_kk = S_new[k, k]
                    v_k = v[k]

                    r_sq = S_kk**2 + v_k**2
                    r = np.sqrt(max(r_sq, eps))

                    if np.abs(S_kk) < eps:
                        S_new[k, k] = r
                        continue

                    c = r / S_kk
                    s = v_k / S_kk
                    S_new[k, k] = r

                    if k < n - 1 and np.abs(c) > eps:
                        S_new[k+1:, k] = (S_new[k+1:, k] + s * v[k+1:]) / c
                        v[k+1:] = c * v[k+1:] - s * S_new[k+1:, k]
            else:
                # Cholesky downdate (rank-1 downdate)
                for k in range(n):
                    S_kk = S_new[k, k]
                    v_k = v[k]

                    r_sq = S_kk**2 - v_k**2
                    if r_sq < eps:
                        r_sq = eps  # 保证正定性
                    r = np.sqrt(r_sq)

                    if np.abs(S_kk) < eps:
                        S_new[k, k] = np.sqrt(eps)
                        continue

                    c = r / S_kk
                    s = v_k / S_kk
                    S_new[k, k] = r

                    if k < n - 1 and np.abs(c) > eps:
                        S_new[k+1:, k] = (S_new[k+1:, k] - s * v[k+1:]) / c
                        v[k+1:] = c * v[k+1:] - s * S_new[k+1:, k]

            # 确保对角元素为正
            for k in range(n):
                if S_new[k, k] < eps:
                    S_new[k, k] = np.sqrt(eps)

        except Exception:
            # 更新失败时，使用直接方法重新计算
            P = S @ S.T + sign * np.outer(v, v)
            P = (P + P.T) / 2
            min_eig = np.min(np.linalg.eigvalsh(P))
            if min_eig < eps:
                P = P + (eps - min_eig) * np.eye(n)
            S_new = np.linalg.cholesky(P)

        return S_new

    def reset(self, initial_soc: float = 0.8):
        """重置滤波器状态"""
        self.x = np.array([initial_soc, 0.0, 0.0])
        P = np.array([
            [1e-4, 0, 0],
            [0, 1e-6, 0],
            [0, 0, 1e-6]
        ])
        self.S = np.linalg.cholesky(P)
        self.R = self.R_init
        self.nu = self.nu_init
        self.step = 0

        if self.ffrls is not None:
            self.ffrls.reset()

        for key in self.history:
            self.history[key] = []

    def predict(self, current: float):
        """
        预测步骤（平方根形式）

        Args:
            current: 电流 (A)
        """
        # 生成Sigma点
        sigma_points = self._generate_sigma_points(self.x, self.S)

        # 传播Sigma点
        n_sigma = sigma_points.shape[1]
        sigma_points_pred = np.zeros_like(sigma_points)

        for i in range(n_sigma):
            sigma_points_pred[:, i] = self.model.state_transition(sigma_points[:, i], current)

        # 计算预测均值
        self.x = np.sum(self.Wm * sigma_points_pred, axis=1)

        # SOC边界约束
        self.x[0] = np.clip(self.x[0], 0, 1)

        # 计算预测协方差平方根 (直接方法，更稳健)
        # 先计算完整的P矩阵，再做Cholesky分解
        P = np.zeros((self.n, self.n))
        for i in range(n_sigma):
            diff = sigma_points_pred[:, i] - self.x
            P += self.Wc[i] * np.outer(diff, diff)
        P += self.Q

        # 确保对称正定
        P = (P + P.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(P))
        if min_eig < 1e-10:
            P = P + (1e-10 - min_eig) * np.eye(self.n)

        try:
            self.S = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            # 如果Cholesky失败，添加更多正则化
            self.S = np.linalg.cholesky(P + 1e-8 * np.eye(self.n))

        self.sigma_points_pred = sigma_points_pred

    def update(self, voltage: float, current: float) -> Tuple[np.ndarray, float, float, float]:
        """
        更新步骤（带NIS门控和Student-t鲁棒更新）

        Args:
            voltage: 测量端电压 (V)
            current: 电流 (A)

        Returns:
            (更新后的状态, 预测电压, NIS值, 鲁棒权重)
        """
        # 重新生成Sigma点
        sigma_points = self._generate_sigma_points(self.x, self.S)

        # 观测预测
        n_sigma = sigma_points.shape[1]
        y_points = np.zeros(n_sigma)
        for i in range(n_sigma):
            y_points[i] = self.model.observation(sigma_points[:, i], current)

        # 预测观测均值
        y_pred = np.sum(self.Wm * y_points)

        # 观测协方差
        S_y = np.sum(self.Wc * (y_points - y_pred)**2) + self.R

        # 交叉协方差
        P_xy = np.zeros(self.n)
        for i in range(n_sigma):
            P_xy += self.Wc[i] * (sigma_points[:, i] - self.x) * (y_points[i] - y_pred)

        # 新息（测量残差）
        e = voltage - y_pred

        # ============ NIS门控 ============
        NIS = e**2 / S_y
        R_effective = self.R

        if self.enable_nis_gate and NIS > self.nis_threshold:
            # 异常测量，放大R降低其影响
            R_effective = self.R * self.nis_R_scale
            S_y = np.sum(self.Wc * (y_points - y_pred)**2) + R_effective

        # ============ Student-t鲁棒更新 ============
        robust_weight = 1.0

        if self.enable_student_t:
            # 自适应调节自由度ν
            if self.adaptive_nu:
                # NIS大 -> ν小（更鲁棒），NIS小 -> ν大（更精确）
                if NIS > 1.0:
                    self.nu = max(self.nu_min, self.nu * 0.95)
                else:
                    self.nu = min(self.nu_max, self.nu * 1.02)

            # Student-t权重计算
            # w = (ν + 1) / (ν + NIS)
            robust_weight = (self.nu + 1) / (self.nu + NIS)
            robust_weight = np.clip(robust_weight, 0.1, 1.0)

            # 调整观测协方差
            S_y_robust = S_y / robust_weight
        else:
            S_y_robust = S_y

        # 卡尔曼增益
        K = P_xy / S_y_robust

        # 状态更新
        self.x = self.x + K * e

        # SOC边界约束
        self.x[0] = np.clip(self.x[0], 0, 1)

        # 协方差平方根更新 (Joseph形式的平方根版本)
        # S_new = cholupdate(S, K * sqrt(S_y), -1)
        try:
            sqrt_S_y = np.sqrt(S_y_robust)
            self.S = self._cholupdate(self.S, K * sqrt_S_y, -1)
        except:
            # 如果downdate失败，使用传统方法重新计算
            P = self.S @ self.S.T - np.outer(K, K) * S_y_robust
            P = (P + P.T) / 2
            min_eig = np.min(np.linalg.eigvalsh(P))
            if min_eig < 1e-10:
                P = P + (1e-10 - min_eig) * np.eye(self.n)
            self.S = np.linalg.cholesky(P)

        # 恢复R（如果被放大过）
        # R会在下一步自动使用原始值

        return self.x, y_pred, NIS, robust_weight

    def estimate_step(
        self,
        voltage: float,
        current: float,
        soc_reference: float = None
    ) -> Dict:
        """
        单步SOC估计

        Args:
            voltage: 测量端电压 (V)
            current: 电流 (A)
            soc_reference: 参考SOC (用于参数辨识，0-100)

        Returns:
            包含估计结果的字典
        """
        self.step += 1

        # 在线参数辨识
        if self.use_online_param_id and self.ffrls is not None:
            soc_for_id = self.x[0] * 100 if soc_reference is None else soc_reference
            params = self.ffrls.identify(voltage, current, soc_for_id, self.step)

            self.model.update_parameters(
                R0=params['R0'],
                R1=params['R1'],
                R2=params['R2'],
                C1=params['C1'],
                C2=params['C2']
            )

        # 预测
        self.predict(current)

        # 更新
        x_updated, Ut_pred, NIS, robust_weight = self.update(voltage, current)

        # 计算误差
        error = voltage - Ut_pred

        # 计算P用于记录 (P = S @ S.T)
        P = self.S @ self.S.T

        # 记录历史
        self.history['SOC'].append(self.x[0])
        self.history['U1'].append(self.x[1])
        self.history['U2'].append(self.x[2])
        self.history['Ut_pred'].append(Ut_pred)
        self.history['error'].append(error)
        self.history['P'].append(P[0, 0])
        self.history['R0'].append(self.model.R0)
        self.history['R1'].append(self.model.R1)
        self.history['R2'].append(self.model.R2)
        self.history['NIS'].append(NIS)
        self.history['nu'].append(self.nu)
        self.history['robust_weight'].append(robust_weight)

        return {
            'SOC': self.x[0],
            'SOC_percent': self.x[0] * 100,
            'U1': self.x[1],
            'U2': self.x[2],
            'Ut_pred': Ut_pred,
            'error': error,
            'P_SOC': P[0, 0],
            'R0': self.model.R0,
            'R1': self.model.R1,
            'R2': self.model.R2,
            'NIS': NIS,
            'nu': self.nu,
            'robust_weight': robust_weight
        }

    def estimate_batch(
        self,
        voltage: np.ndarray,
        current: np.ndarray,
        soc_reference: np.ndarray = None,
        initial_soc: float = None
    ) -> Dict:
        """
        批量SOC估计

        Args:
            voltage: 电压序列 (V)
            current: 电流序列 (A)
            soc_reference: 参考SOC序列 (0-100)
            initial_soc: 初始SOC (0-1)

        Returns:
            包含所有估计结果的字典
        """
        if initial_soc is not None:
            self.reset(initial_soc)

        n = len(voltage)
        results = {
            'SOC': np.zeros(n),
            'SOC_percent': np.zeros(n),
            'U1': np.zeros(n),
            'U2': np.zeros(n),
            'Ut_pred': np.zeros(n),
            'error': np.zeros(n),
            'P_SOC': np.zeros(n),
            'R0': np.zeros(n),
            'R1': np.zeros(n),
            'R2': np.zeros(n),
            'NIS': np.zeros(n),
            'nu': np.zeros(n),
            'robust_weight': np.zeros(n)
        }

        for i in range(n):
            soc_ref = soc_reference[i] if soc_reference is not None else None
            result = self.estimate_step(voltage[i], current[i], soc_ref)

            for key in results:
                results[key][i] = result[key]

        return results


# 测试代码
if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    # 加载测试数据
    data_path = Path(__file__).parent.parent.parent / "dataset" / "processed" / "25C_DST_80SOC.csv"

    if data_path.exists():
        print(f"Loading test data from: {data_path}")
        df = pd.read_csv(data_path)

        voltage = df['voltage_V'].values
        current = df['current_A'].values
        soc_true = df['soc_percent'].values
        time = df['time_s'].values

        # 创建Robust SR-UKF估计器
        initial_soc = soc_true[0] / 100
        sr_ukf = RobustSRUKF(
            initial_soc=initial_soc,
            capacity_Ah=2.0,
            sample_time=1.0,
            use_online_param_id=True,
            enable_nis_gate=True,
            enable_student_t=True,
            adaptive_nu=True
        )

        # 批量估计
        print("Running Robust SR-UKF estimation...")
        results = sr_ukf.estimate_batch(voltage, current, soc_true, initial_soc)

        # 计算误差
        soc_error = results['SOC_percent'] - soc_true
        rmse = np.sqrt(np.mean(soc_error**2))
        mae = np.mean(np.abs(soc_error))
        max_error = np.max(np.abs(soc_error))

        print(f"\nRobust SR-UKF Estimation Results:")
        print(f"  RMSE: {rmse:.4f}%")
        print(f"  MAE: {mae:.4f}%")
        print(f"  Max Error: {max_error:.4f}%")

        # 统计NIS门控触发次数
        nis_triggered = np.sum(results['NIS'] > 6.63)
        print(f"  NIS Gate Triggered: {nis_triggered} times ({100*nis_triggered/len(soc_true):.2f}%)")

        # 绘制结果
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))

        # SOC对比
        axes[0].plot(time, soc_true, 'b-', linewidth=1, label='True SOC')
        axes[0].plot(time, results['SOC_percent'], 'r--', linewidth=1, label='SR-UKF Estimated')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('SOC (%)')
        axes[0].set_title('Robust SR-UKF SOC Estimation')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # SOC误差
        axes[1].plot(time, soc_error, 'g-', linewidth=0.8)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('SOC Error (%)')
        axes[1].set_title(f'SOC Estimation Error (RMSE={rmse:.4f}%)')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        # NIS和鲁棒权重
        ax2_twin = axes[2].twinx()
        axes[2].plot(time, results['NIS'], 'b-', linewidth=0.5, alpha=0.7, label='NIS')
        axes[2].axhline(y=6.63, color='r', linestyle='--', linewidth=1, label='NIS Threshold')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('NIS', color='b')
        axes[2].set_ylim([0, min(20, np.max(results['NIS']) * 1.1)])
        axes[2].legend(loc='upper left')

        ax2_twin.plot(time, results['robust_weight'], 'g-', linewidth=0.5, alpha=0.7, label='Robust Weight')
        ax2_twin.set_ylabel('Robust Weight', color='g')
        ax2_twin.set_ylim([0, 1.1])
        ax2_twin.legend(loc='upper right')

        axes[2].set_title('NIS and Robust Weight')
        axes[2].grid(True, alpha=0.3)

        # 自适应自由度
        axes[3].plot(time, results['nu'], 'm-', linewidth=0.8)
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Student-t ν')
        axes[3].set_title('Adaptive Degrees of Freedom')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = data_path.parent / "RobustSRUKF_results.png"
        plt.savefig(save_path, dpi=150)
        print(f"\nResults saved to: {save_path}")
        plt.close()

    else:
        print(f"Test data not found: {data_path}")
        print("Please run process_battery_data.py first.")
