"""
UKF (Unscented Kalman Filter) SOC Estimation
无迹卡尔曼滤波SOC估计算法

基于二阶RC等效电路模型，使用Sigma点采样近似非线性变换。

UKF优势:
    - 不需要计算雅可比矩阵
    - 对强非线性系统精度更高（如OCV-SOC曲线）
    - 实现相对简单

State Vector: x = [SOC, U1, U2]^T
    - SOC: State of Charge (0-1)
    - U1: First RC circuit polarization voltage (V)
    - U2: Second RC circuit polarization voltage (V)

Reference:
    - Julier, S. J., & Uhlmann, J. K. (2004). Unscented filtering and nonlinear estimation
    - Plett, G. L. (2006). Sigma-point Kalman filtering for battery management systems

Author: Auto-generated for CALCE dataset
Date: 2024
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from .battery_model import BatteryModel2RC
from .ffrls import FFRLS


class UKF:
    """
    无迹卡尔曼滤波器

    用于电池SOC估计，使用Sigma点采样处理非线性。
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
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0
    ):
        """
        初始化UKF估计器

        Args:
            initial_soc: 初始SOC (0-1)
            capacity_Ah: 电池容量 (Ah)
            sample_time: 采样时间 (s)
            process_noise: 过程噪声协方差矩阵Q (3x3)
            measurement_noise: 测量噪声协方差R (标量)
            initial_covariance: 初始状态协方差矩阵P (3x3)
            use_online_param_id: 是否使用在线参数辨识
            alpha: Sigma点分布参数，控制分布范围 (1e-4 ~ 1)
            beta: 分布类型参数，高斯分布时beta=2最优
            kappa: 次要缩放参数，通常为0或3-n
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
                [1e-6, 0, 0],      # SOC过程噪声
                [0, 1e-8, 0],      # U1过程噪声
                [0, 0, 1e-8]       # U2过程噪声
            ])
        else:
            self.Q = process_noise.copy()

        # 测量噪声协方差 R
        self.R = measurement_noise

        # 状态协方差矩阵 P
        if initial_covariance is None:
            self.P = np.array([
                [1e-4, 0, 0],
                [0, 1e-6, 0],
                [0, 0, 1e-6]
            ])
        else:
            self.P = initial_covariance.copy()

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
            'R2': []
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

    def _generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        生成Sigma点

        Args:
            x: 状态均值 (n,)
            P: 状态协方差 (n, n)

        Returns:
            sigma_points: Sigma点矩阵 (n, 2n+1)
        """
        n = self.n
        lambda_ = self.lambda_

        # 计算矩阵平方根
        try:
            sqrt_P = np.linalg.cholesky((n + lambda_) * P)
        except np.linalg.LinAlgError:
            # 如果Cholesky分解失败，添加小量对角项
            sqrt_P = np.linalg.cholesky((n + lambda_) * P + 1e-10 * np.eye(n))

        # 生成Sigma点
        sigma_points = np.zeros((n, 2 * n + 1))
        sigma_points[:, 0] = x

        for i in range(n):
            sigma_points[:, i + 1] = x + sqrt_P[:, i]
            sigma_points[:, n + i + 1] = x - sqrt_P[:, i]

        return sigma_points

    def _unscented_transform(
        self,
        sigma_points: np.ndarray,
        func,
        noise_cov: np.ndarray,
        *func_args
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        无迹变换

        Args:
            sigma_points: Sigma点矩阵 (n_in, 2n+1)
            func: 非线性函数
            noise_cov: 噪声协方差矩阵
            *func_args: 函数额外参数

        Returns:
            y_mean: 变换后的均值
            y_cov: 变换后的协方差
            y_points: 变换后的Sigma点
        """
        n_sigma = sigma_points.shape[1]

        # 通过非线性函数传播Sigma点
        y_points = []
        for i in range(n_sigma):
            y_i = func(sigma_points[:, i], *func_args)
            y_points.append(y_i)

        y_points = np.array(y_points).T  # (n_out, 2n+1)

        # 计算加权均值
        if y_points.ndim == 1:
            y_mean = np.sum(self.Wm * y_points)
        else:
            y_mean = np.sum(self.Wm * y_points, axis=1)

        # 计算加权协方差
        if y_points.ndim == 1:
            y_cov = np.sum(self.Wc * (y_points - y_mean)**2) + noise_cov
        else:
            y_diff = y_points - y_mean.reshape(-1, 1)
            y_cov = np.zeros((y_points.shape[0], y_points.shape[0]))
            for i in range(n_sigma):
                y_cov += self.Wc[i] * np.outer(y_diff[:, i], y_diff[:, i])
            y_cov += noise_cov

        return y_mean, y_cov, y_points

    def reset(self, initial_soc: float = 0.8):
        """重置滤波器状态"""
        self.x = np.array([initial_soc, 0.0, 0.0])
        self.P = np.array([
            [1e-4, 0, 0],
            [0, 1e-6, 0],
            [0, 0, 1e-6]
        ])
        self.step = 0

        if self.ffrls is not None:
            self.ffrls.reset()

        for key in self.history:
            self.history[key] = []

    def predict(self, current: float):
        """
        预测步骤

        Args:
            current: 电流 (A)，放电为负，充电为正
        """
        # 生成Sigma点
        sigma_points = self._generate_sigma_points(self.x, self.P)

        # 状态转移函数包装
        def state_func(x, I):
            return self.model.state_transition(x, I)

        # 无迹变换预测
        self.x, self.P, self.sigma_points_pred = self._unscented_transform(
            sigma_points, state_func, self.Q, current
        )

        # SOC边界约束
        self.x[0] = np.clip(self.x[0], 0, 1)

    def update(self, voltage: float, current: float) -> Tuple[np.ndarray, float]:
        """
        更新步骤

        Args:
            voltage: 测量端电压 (V)
            current: 电流 (A)

        Returns:
            (更新后的状态向量, 预测电压)
        """
        # 重新生成预测后的Sigma点
        sigma_points = self._generate_sigma_points(self.x, self.P)

        # 观测函数包装
        def obs_func(x, I):
            return self.model.observation(x, I)

        # 通过观测函数传播Sigma点
        n_sigma = sigma_points.shape[1]
        y_points = np.zeros(n_sigma)
        for i in range(n_sigma):
            y_points[i] = obs_func(sigma_points[:, i], current)

        # 预测观测均值
        y_pred = np.sum(self.Wm * y_points)

        # 观测协方差
        Pyy = np.sum(self.Wc * (y_points - y_pred)**2) + self.R

        # 交叉协方差
        Pxy = np.zeros(self.n)
        for i in range(n_sigma):
            Pxy += self.Wc[i] * (sigma_points[:, i] - self.x) * (y_points[i] - y_pred)

        # 卡尔曼增益
        K = Pxy / Pyy

        # 测量残差
        y = voltage - y_pred

        # 状态更新
        self.x = self.x + K * y

        # SOC边界约束
        self.x[0] = np.clip(self.x[0], 0, 1)

        # 协方差更新
        self.P = self.P - np.outer(K, K) * Pyy

        # 确保协方差矩阵对称和正定
        self.P = (self.P + self.P.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(self.P))
        if min_eig < 1e-10:
            self.P = self.P + (1e-10 - min_eig) * np.eye(self.n)

        return self.x, y_pred

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
            current: 电流 (A)，放电为负，充电为正
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
        x_updated, Ut_pred = self.update(voltage, current)

        # 计算误差
        error = voltage - Ut_pred

        # 记录历史
        self.history['SOC'].append(self.x[0])
        self.history['U1'].append(self.x[1])
        self.history['U2'].append(self.x[2])
        self.history['Ut_pred'].append(Ut_pred)
        self.history['error'].append(error)
        self.history['P'].append(self.P[0, 0])
        self.history['R0'].append(self.model.R0)
        self.history['R1'].append(self.model.R1)
        self.history['R2'].append(self.model.R2)

        return {
            'SOC': self.x[0],
            'SOC_percent': self.x[0] * 100,
            'U1': self.x[1],
            'U2': self.x[2],
            'Ut_pred': Ut_pred,
            'error': error,
            'P_SOC': self.P[0, 0],
            'R0': self.model.R0,
            'R1': self.model.R1,
            'R2': self.model.R2
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
            'R2': np.zeros(n)
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

        # 创建UKF估计器
        initial_soc = soc_true[0] / 100
        ukf = UKF(
            initial_soc=initial_soc,
            capacity_Ah=2.0,
            sample_time=1.0,
            use_online_param_id=True
        )

        # 批量估计
        print("Running UKF estimation...")
        results = ukf.estimate_batch(voltage, current, soc_true, initial_soc)

        # 计算误差
        soc_error = results['SOC_percent'] - soc_true
        rmse = np.sqrt(np.mean(soc_error**2))
        mae = np.mean(np.abs(soc_error))
        max_error = np.max(np.abs(soc_error))

        print(f"\nEstimation Results:")
        print(f"  RMSE: {rmse:.4f}%")
        print(f"  MAE: {mae:.4f}%")
        print(f"  Max Error: {max_error:.4f}%")

        # 绘制结果
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # SOC对比
        axes[0].plot(time, soc_true, 'b-', linewidth=1, label='True SOC')
        axes[0].plot(time, results['SOC_percent'], 'r--', linewidth=1, label='Estimated SOC')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('SOC (%)')
        axes[0].set_title('UKF SOC Estimation')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # SOC误差
        axes[1].plot(time, soc_error, 'g-', linewidth=0.8)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('SOC Error (%)')
        axes[1].set_title(f'SOC Estimation Error (RMSE={rmse:.4f}%)')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        # 电压对比
        axes[2].plot(time, voltage, 'b-', linewidth=0.8, label='Measured')
        axes[2].plot(time, results['Ut_pred'], 'r--', linewidth=0.8, label='Predicted')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Voltage (V)')
        axes[2].set_title('Terminal Voltage')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = data_path.parent / "UKF_results.png"
        plt.savefig(save_path, dpi=150)
        print(f"\nResults saved to: {save_path}")
        plt.close()

    else:
        print(f"Test data not found: {data_path}")
        print("Please run process_battery_data.py first.")
