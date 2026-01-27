"""
AEKF (Adaptive Extended Kalman Filter) SOC Estimation
自适应扩展卡尔曼滤波SOC估计算法

基于二阶RC等效电路模型，使用自适应过程噪声协方差。

State Vector: x = [SOC, U1, U2]^T
    - SOC: State of Charge (0-1)
    - U1: First RC circuit polarization voltage (V)
    - U2: Second RC circuit polarization voltage (V)

Reference:
    - Plett, G. L. (2004). Extended Kalman filtering for battery management systems
    - Adaptive noise covariance based on innovation sequence

Author: Auto-generated for CALCE dataset
Date: 2024
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from .battery_model import BatteryModel2RC
from .ffrls import FFRLS


class AEKF:
    """
    自适应扩展卡尔曼滤波器

    用于电池SOC估计，支持在线参数辨识。
    """

    def __init__(
        self,
        initial_soc: float = 0.6,
        capacity_Ah: float = 2.0,
        sample_time: float = 1.0,
        process_noise: np.ndarray = None,
        measurement_noise: float = 2e-4,
        initial_covariance: np.ndarray = None,
        use_online_param_id: bool = True,
        adaptive_Q: bool = True,
        temperature: Optional[str] = None
    ):
        """
        初始化AEKF估计器

        Args:
            initial_soc: 初始SOC (0-1)
            capacity_Ah: 电池容量 (Ah)
            sample_time: 采样时间 (s)
            process_noise: 过程噪声协方差矩阵Q (3x3)
            measurement_noise: 测量噪声协方差R (标量)
            initial_covariance: 初始状态协方差矩阵P (3x3)
            use_online_param_id: 是否使用在线参数辨识
            adaptive_Q: 是否使用自适应过程噪声
            temperature: 温度标签 ("0C", "25C", "45C")，用于自动选择OCV系数，必须指定
        """
        if temperature is None:
            raise ValueError("Must specify 'temperature' parameter")
        # 状态维度
        self.n_states = 3

        # 初始状态 [SOC, U1, U2]
        self.x = np.array([initial_soc, 0.0, 0.0])

        # 采样时间
        self.dt = sample_time

        # 过程噪声协方差矩阵 Q
        if process_noise is None:
            self.Q = np.array([
                [1e-6, 0, 0],      # SOC过程噪声
                [0, 1e-8, 0],      # U1过程噪声
                [0, 0, 1e-8]       # U2过程噪声
            ])
        else:
            self.Q = process_noise.copy()

        self.Q_init = self.Q.copy()

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

        # 温度标签
        self.temperature = temperature

        # 电池模型
        self.model = BatteryModel2RC(
            capacity_Ah=capacity_Ah,
            sample_time=sample_time,
            temperature=temperature
        )

        # 在线参数辨识器
        self.use_online_param_id = use_online_param_id
        if use_online_param_id:
            self.ffrls = FFRLS(sample_time=sample_time, temperature=temperature)
        else:
            self.ffrls = None

        # 自适应Q标志
        self.adaptive_Q = adaptive_Q

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
            'K': [],
            'R0': [],
            'R1': [],
            'R2': []
        }

    def reset(self, initial_soc: float = 0.8):
        """重置滤波器状态"""
        self.x = np.array([initial_soc, 0.0, 0.0])
        self.P = np.array([
            [1e-4, 0, 0],
            [0, 1e-6, 0],
            [0, 0, 1e-6]
        ])
        self.Q = self.Q_init.copy()
        self.step = 0

        if self.ffrls is not None:
            self.ffrls.reset()

        for key in self.history:
            self.history[key] = []

    def predict(self, current: float) -> np.ndarray:
        """
        预测步骤

        Args:
            current: 电流 (A)，放电为负，充电为正

        Returns:
            预测的状态向量
        """
        # 状态转移
        self.x = self.model.state_transition(self.x, current)

        # 获取状态雅可比矩阵
        A = self.model.get_state_jacobian(self.x, current)

        # 协方差预测
        self.P = A @ self.P @ A.T + self.Q

        return self.x

    def update(self, voltage: float, current: float) -> Tuple[np.ndarray, float]:
        """
        更新步骤

        Args:
            voltage: 测量端电压 (V)
            current: 电流 (A)

        Returns:
            (更新后的状态向量, 预测电压)
        """
        # 预测端电压
        Ut_pred = self.model.observation(self.x, current)

        # 测量残差（新息）
        y = voltage - Ut_pred

        # 获取观测雅可比矩阵
        C = self.model.get_observation_jacobian(self.x, current)

        # 计算卡尔曼增益
        S = C @ self.P @ C.T + self.R  # 新息协方差（预测阶段）
        S_prior = S  # 保存预测阶段的S，用于后续Q自适应
        K = self.P @ C.T / S

        # 状态更新
        self.x = self.x + K * y

        # SOC边界约束
        self.x[0] = np.clip(self.x[0], 0, 1)

        # 协方差更新 - 使用Joseph形式（数值稳定）
        # P = (I - KC) @ P @ (I - KC)^T + K @ R @ K^T
        I_KC = np.eye(self.n_states) - np.outer(K, C)
        self.P = I_KC @ self.P @ I_KC.T + np.outer(K, K) * self.R

        # 确保协方差矩阵对称和正定
        self.P = (self.P + self.P.T) / 2

        # 自适应过程噪声协方差更新
        if self.adaptive_Q:
            # 基于新息的自适应方法（使用指数加权平滑）
            alpha_Q = 0.98  # 平滑因子，避免Q剧烈变化
            # 使用预测阶段的S_prior（更新前），而非更新后的P重新计算
            # 这样保持时序一致性：用预测时的统计量来调整Q
            Q_innovation = np.outer(K, K) * S_prior  # 使用预测阶段的新息协方差
            self.Q = alpha_Q * self.Q + (1 - alpha_Q) * Q_innovation
            # 添加上下界约束，防止Q过大或过小导致发散
            self.Q = np.clip(self.Q, self.Q_init * 0.1, self.Q_init * 10)
            # 保持对角占优，提高数值稳定性
            self.Q = np.diag(np.diag(self.Q))

        return self.x, Ut_pred

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
            # 使用当前SOC估计值进行参数辨识
            soc_for_id = self.x[0] * 100 if soc_reference is None else soc_reference
            params = self.ffrls.identify(voltage, current, soc_for_id, self.step)

            # 更新模型参数
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
        self.history['P'].append(self.P[0, 0])  # SOC方差
        self.history['K'].append(self.P @ self.model.get_observation_jacobian(self.x, current).T /
                                 (self.model.get_observation_jacobian(self.x, current) @ self.P @
                                  self.model.get_observation_jacobian(self.x, current).T + self.R))
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
            soc_reference: 参考SOC序列 (0-100)，用于参数辨识
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

        # 创建AEKF估计器
        initial_soc = soc_true[0] / 100  # 使用真实初始SOC
        aekf = AEKF(
            initial_soc=initial_soc,
            capacity_Ah=2.0,
            sample_time=1.0,
            use_online_param_id=True,
            adaptive_Q=True
        )

        # 批量估计
        print("Running AEKF estimation...")
        results = aekf.estimate_batch(voltage, current, soc_true, initial_soc)

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
        axes[0].set_title('AEKF SOC Estimation')
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
        save_path = data_path.parent / "AEKF_results.png"
        plt.savefig(save_path, dpi=150)
        print(f"\nResults saved to: {save_path}")
        plt.close()

    else:
        print(f"Test data not found: {data_path}")
        print("Please run process_battery_data.py first.")
