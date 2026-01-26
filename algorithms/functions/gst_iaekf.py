"""
GST-IAEKF (Gated Strong-Tracking Innovation-Adaptive EKF)
门控强跟踪创新自适应扩展卡尔曼滤波SOC估计算法

创新点：
    1. NIS门控层: 基于归一化新息平方检测异常测量
    2. 强跟踪/遗忘因子: 引入λ因子放大预测协方差，提高对工况突变的响应
    3. 滑窗残差Q/R自适应: 基于滑动窗口新息协方差估计，自动调节Q和R

核心思想：
    - 门控层: "硬保护"，异常时跳过更新或增大R
    - 强跟踪: "快响应"，工况突变时放大P让滤波器跟得上
    - Q/R自适应: "软调节"，用残差统计量自动修正噪声协方差

State Vector: x = [SOC, U1, U2]^T

Reference:
    - Zhou, D. H., & Frank, P. M. (1996). Strong tracking Kalman filter
    - Mehra, R. K. (1972). Approaches to adaptive filtering
    - Mohamed, A. H., & Schwarz, K. P. (1999). Adaptive Kalman filtering

Author: Generated for CALCE dataset
Date: 2024
"""

import numpy as np
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple
from .battery_model import BatteryModel2RC
from .ffrls import FFRLS


class GSTIAEKF:
    """
    门控强跟踪创新自适应扩展卡尔曼滤波器

    结合NIS门控、强跟踪因子和滑窗Q/R自适应，
    实现轻量化、鲁棒的SOC估计。
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
        # NIS门控参数
        enable_nis_gate: bool = True,
        nis_threshold: float = 6.63,      # χ²(1, 0.99)
        nis_R_scale: float = 10.0,        # 异常时R放大倍数
        # 强跟踪参数
        enable_strong_tracking: bool = True,
        lambda_min: float = 1.0,          # 最小遗忘因子
        lambda_max: float = 10.0,         # 最大遗忘因子
        rho: float = 0.95,                # 遗忘因子衰减率
        # Q/R自适应参数
        enable_qr_adaptive: bool = True,
        window_size: int = 10,            # 滑窗大小
        qr_alpha: float = 0.98,           # Q/R更新平滑因子
    ):
        """
        初始化GST-IAEKF估计器

        Args:
            initial_soc: 初始SOC (0-1)
            capacity_Ah: 电池容量 (Ah)
            sample_time: 采样时间 (s)
            process_noise: 过程噪声协方差矩阵Q (3x3)
            measurement_noise: 测量噪声协方差R (标量)
            initial_covariance: 初始状态协方差矩阵P (3x3)
            use_online_param_id: 是否使用在线参数辨识
            enable_nis_gate: 是否启用NIS门控
            nis_threshold: NIS门控阈值
            nis_R_scale: NIS超阈值时R的放大倍数
            enable_strong_tracking: 是否启用强跟踪
            lambda_min: 最小强跟踪因子
            lambda_max: 最大强跟踪因子
            rho: 强跟踪因子衰减率
            enable_qr_adaptive: 是否启用Q/R自适应
            window_size: 滑窗大小
            qr_alpha: Q/R更新平滑因子
        """
        # 状态维度
        self.n = 3

        # 初始状态 [SOC, U1, U2]
        self.x = np.array([initial_soc, 0.0, 0.0])

        # 采样时间
        self.dt = sample_time

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

        # 测量噪声协方差 R
        self.R = measurement_noise
        self.R_init = measurement_noise

        # 状态协方差矩阵 P
        if initial_covariance is None:
            self.P = np.array([
                [1e-4, 0, 0],
                [0, 1e-6, 0],
                [0, 0, 1e-6]
            ])
        else:
            self.P = initial_covariance.copy()
        self.P_init = self.P.copy()

        # NIS门控参数
        self.enable_nis_gate = enable_nis_gate
        self.nis_threshold = nis_threshold
        self.nis_R_scale = nis_R_scale

        # 强跟踪参数
        self.enable_strong_tracking = enable_strong_tracking
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.rho = rho
        self.lambda_k = 1.0  # 当前强跟踪因子

        # Q/R自适应参数
        self.enable_qr_adaptive = enable_qr_adaptive
        self.window_size = window_size
        self.qr_alpha = qr_alpha
        self.innovation_window = deque(maxlen=window_size)  # 新息滑窗

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
            'lambda': [],
            'Q_SOC': [],
            'R_adaptive': [],
            'gate_triggered': []
        }

    def reset(self, initial_soc: float = 0.8):
        """重置滤波器状态"""
        self.x = np.array([initial_soc, 0.0, 0.0])
        self.P = self.P_init.copy()
        self.Q = self.Q_init.copy()
        self.R = self.R_init
        self.lambda_k = 1.0
        self.innovation_window.clear()
        self.step = 0

        if self.ffrls is not None:
            self.ffrls.reset()

        for key in self.history:
            self.history[key] = []

    def predict(self, current: float):
        """
        预测步骤

        Args:
            current: 电流 (A)
        """
        # 状态转移
        self.x = self.model.state_transition(self.x, current)

        # SOC边界约束
        self.x[0] = np.clip(self.x[0], 0, 1)

        # 状态转移雅可比矩阵
        A = self.model.get_state_jacobian(self.x, current)

        # 预测协方差
        self.P = A @ self.P @ A.T + self.Q

        # ============ 强跟踪因子 ============
        if self.enable_strong_tracking and self.lambda_k > 1.0:
            # 放大预测协方差，提高对突变的响应
            self.P = self.lambda_k * self.P
            # 遗忘因子衰减
            self.lambda_k = max(self.lambda_min, self.lambda_k * self.rho)

        # 确保对称正定
        self.P = (self.P + self.P.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(self.P))
        if min_eig < 1e-10:
            self.P = self.P + (1e-10 - min_eig) * np.eye(self.n)

    def update(self, voltage: float, current: float) -> Tuple[np.ndarray, float, float, bool]:
        """
        更新步骤（含门控、强跟踪、Q/R自适应）

        Args:
            voltage: 测量端电压 (V)
            current: 电流 (A)

        Returns:
            (更新后的状态, 预测电压, NIS值, 门控是否触发)
        """
        # 预测观测值
        Ut_pred = self.model.observation(self.x, current)

        # 观测雅可比矩阵
        C = self.model.get_observation_jacobian(self.x, current)

        # 新息（测量残差）
        e = voltage - Ut_pred

        # 新息协方差
        S = C @ self.P @ C.T + self.R

        # ============ NIS计算 ============
        NIS = e**2 / S
        gate_triggered = False
        R_effective = self.R

        # ============ NIS门控层 ============
        if self.enable_nis_gate and NIS > self.nis_threshold:
            gate_triggered = True
            # 异常测量：放大R降低其影响
            R_effective = self.R * self.nis_R_scale
            S = C @ self.P @ C.T + R_effective

            # 同时激活强跟踪（可能是工况突变）
            if self.enable_strong_tracking:
                # 根据NIS大小调整λ
                self.lambda_k = min(self.lambda_max, 1.0 + (NIS - self.nis_threshold) * 0.5)

        # ============ 滑窗Q/R自适应 ============
        if self.enable_qr_adaptive:
            # 将新息加入滑窗
            self.innovation_window.append(e)

            if len(self.innovation_window) >= self.window_size:
                # 计算滑窗新息协方差估计
                innovations = np.array(self.innovation_window)
                H_k = np.mean(innovations ** 2)  # 新息方差估计

                # 更新R (基于新息协方差)
                # 理论上: E[e²] = C*P*C' + R，所以 R ≈ H_k - C*P*C'
                CPCt = C @ self.P @ C.T
                R_estimate = max(H_k - CPCt, self.R_init * 0.1)

                # 平滑更新R
                if not gate_triggered:  # 只在正常时更新
                    self.R = self.qr_alpha * self.R + (1 - self.qr_alpha) * R_estimate
                    # 约束R在合理范围
                    self.R = np.clip(self.R, self.R_init * 0.1, self.R_init * 10)

                # 更新Q (基于残差统计)
                # 使用简化方法：Q与新息方差成正比
                if not gate_triggered:
                    Q_scale = H_k / (self.R_init + CPCt) if (self.R_init + CPCt) > 0 else 1.0
                    Q_scale = np.clip(Q_scale, 0.5, 2.0)
                    self.Q = self.qr_alpha * self.Q + (1 - self.qr_alpha) * self.Q_init * Q_scale

        # ============ 卡尔曼增益 ============
        K = self.P @ C.T / S

        # ============ 状态更新 ============
        self.x = self.x + K * e

        # SOC边界约束
        self.x[0] = np.clip(self.x[0], 0, 1)

        # ============ 协方差更新 (Joseph形式，更稳定) ============
        I_KC = np.eye(self.n) - np.outer(K, C)
        self.P = I_KC @ self.P @ I_KC.T + np.outer(K, K) * R_effective

        # 确保对称正定
        self.P = (self.P + self.P.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(self.P))
        if min_eig < 1e-10:
            self.P = self.P + (1e-10 - min_eig) * np.eye(self.n)

        return self.x, Ut_pred, NIS, gate_triggered

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
        x_updated, Ut_pred, NIS, gate_triggered = self.update(voltage, current)

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
        self.history['NIS'].append(NIS)
        self.history['lambda'].append(self.lambda_k)
        self.history['Q_SOC'].append(self.Q[0, 0])
        self.history['R_adaptive'].append(self.R)
        self.history['gate_triggered'].append(gate_triggered)

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
            'R2': self.model.R2,
            'NIS': NIS,
            'lambda': self.lambda_k,
            'Q_SOC': self.Q[0, 0],
            'R_adaptive': self.R,
            'gate_triggered': gate_triggered
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
            'lambda': np.zeros(n),
            'Q_SOC': np.zeros(n),
            'R_adaptive': np.zeros(n),
            'gate_triggered': np.zeros(n, dtype=bool)
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

        # 过滤SOC范围 (80% -> 10%)
        mask = df['soc_percent'] >= 10.0
        df = df[mask].reset_index(drop=True)

        voltage = df['voltage_V'].values
        current = df['current_A'].values
        soc_true = df['soc_percent'].values
        time = df['time_s'].values

        print(f"Data points: {len(voltage)} (SOC: {soc_true.min():.1f}% ~ {soc_true.max():.1f}%)")

        # 创建GST-IAEKF估计器
        initial_soc = soc_true[0] / 100
        gst_iaekf = GSTIAEKF(
            initial_soc=initial_soc,
            capacity_Ah=2.0,
            sample_time=1.0,
            use_online_param_id=True,
            enable_nis_gate=True,
            enable_strong_tracking=True,
            enable_qr_adaptive=True,
            window_size=10
        )

        # 批量估计
        print("Running GST-IAEKF estimation...")
        results = gst_iaekf.estimate_batch(voltage, current, soc_true, initial_soc)

        # 计算误差
        soc_error = results['SOC_percent'] - soc_true
        rmse = np.sqrt(np.mean(soc_error**2))
        mae = np.mean(np.abs(soc_error))
        max_error = np.max(np.abs(soc_error))

        # 统计
        gate_count = np.sum(results['gate_triggered'])
        print(f"\nGST-IAEKF Estimation Results:")
        print(f"  RMSE: {rmse:.4f}%")
        print(f"  MAE: {mae:.4f}%")
        print(f"  Max Error: {max_error:.4f}%")
        print(f"  NIS Gate Triggered: {gate_count} times ({100*gate_count/len(soc_true):.2f}%)")

        # 绘制结果
        fig, axes = plt.subplots(4, 1, figsize=(14, 14))

        # SOC对比
        axes[0].plot(time, soc_true, 'b-', linewidth=1.5, label='True SOC')
        axes[0].plot(time, results['SOC_percent'], 'r--', linewidth=1.5, label='GST-IAEKF Estimated')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('SOC (%)')
        axes[0].set_title(f'GST-IAEKF SOC Estimation (RMSE={rmse:.4f}%)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # SOC误差
        axes[1].plot(time, soc_error, 'g-', linewidth=0.8)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('SOC Error (%)')
        axes[1].set_title(f'Estimation Error (Max={max_error:.4f}%)')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        # NIS和强跟踪因子
        ax2_twin = axes[2].twinx()
        axes[2].plot(time, results['NIS'], 'b-', linewidth=0.5, alpha=0.7, label='NIS')
        axes[2].axhline(y=6.63, color='r', linestyle='--', linewidth=1, label='Threshold')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('NIS', color='b')
        axes[2].set_ylim([0, min(15, np.max(results['NIS']) * 1.1)])
        axes[2].legend(loc='upper left')

        ax2_twin.plot(time, results['lambda'], 'g-', linewidth=1, alpha=0.8, label='λ (Strong Tracking)')
        ax2_twin.set_ylabel('λ', color='g')
        ax2_twin.set_ylim([0.9, max(1.5, np.max(results['lambda']) * 1.1)])
        ax2_twin.legend(loc='upper right')

        axes[2].set_title('NIS and Strong Tracking Factor')
        axes[2].grid(True, alpha=0.3)

        # 自适应Q和R
        ax3_twin = axes[3].twinx()
        axes[3].plot(time, results['Q_SOC'] * 1e6, 'm-', linewidth=0.8, label='Q_SOC (×1e-6)')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Q_SOC (×1e-6)', color='m')
        axes[3].legend(loc='upper left')

        ax3_twin.plot(time, results['R_adaptive'] * 1e4, 'c-', linewidth=0.8, label='R (×1e-4)')
        ax3_twin.set_ylabel('R (×1e-4)', color='c')
        ax3_twin.legend(loc='upper right')

        axes[3].set_title('Adaptive Q and R')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = data_path.parent / "GSTIAEKF_results.png"
        plt.savefig(save_path, dpi=150)
        print(f"\nResults saved to: {save_path}")
        plt.close()

    else:
        print(f"Test data not found: {data_path}")
        print("Please run process_battery_data.py first.")
