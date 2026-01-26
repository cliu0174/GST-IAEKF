"""
FFRLS (Forgetting Factor Recursive Least Squares) Parameter Identification
遗忘因子递推最小二乘参数辨识算法

基于二阶RC等效电路模型 (2RC Thevenin Model):

    Uoc ----[ R0 ]----+----[ R1 ]----+----[ R2 ]----+---- Ut
                      |              |              |
                     [C1]          [C2]            |
                      |              |              |
                      +--------------+--------------+

模型方程:
    Ut = Uoc - I*R0 - U1 - U2
    dU1/dt = I/C1 - U1/(R1*C1)
    dU2/dt = I/C2 - U2/(R2*C2)

离散化后的传递函数形式用于参数辨识。

Reference:
    - Plett, G. L. (2004). Extended Kalman filtering for battery management systems
    - 递推最小二乘法在锂电池参数辨识中的应用

Author: Auto-generated based on reference implementation
Date: 2024
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union


class FFRLS:
    """
    FFRLS参数辨识器（二阶RC模型）

    Forgetting Factor Recursive Least Squares for 2nd order RC model
    parameter identification.

    Attributes:
        R0: 欧姆内阻 (Ohm)
        R1: 第一RC环极化内阻 (Ohm)
        R2: 第二RC环极化内阻 (Ohm)
        C1: 第一RC环极化电容 (F)
        C2: 第二RC环极化电容 (F)
        tau1: 第一时间常数 R1*C1 (s)
        tau2: 第二时间常数 R2*C2 (s)
    """

    def __init__(
        self,
        sample_time: float = 1.0,
        forgetting_factor: float = 0.998,
        initial_covariance: float = 10.0,
        sigma: float = 0.001,
        ocv_soc_coeffs: np.ndarray = None,
        adaptive_lambda: bool = True,
        min_current_threshold: float = 0.001
    ):
        """
        初始化FFRLS参数辨识器

        Args:
            sample_time: 采样时间 (s)
            forgetting_factor: 初始遗忘因子 (0.95-0.999)
            initial_covariance: 协方差矩阵初始值
            sigma: 自适应遗忘因子的参数
            ocv_soc_coeffs: OCV-SOC多项式系数 (高阶在前)，如果为None则使用默认值
            adaptive_lambda: 是否使用自适应遗忘因子
            min_current_threshold: 最小电流阈值，低于此值不进行参数更新 (A)
        """
        self.T = sample_time
        self.lambda_init = forgetting_factor
        self.lamda = forgetting_factor  # 当前遗忘因子
        self.sigma = sigma
        self.adaptive_lambda = adaptive_lambda
        self.min_current_threshold = min_current_threshold

        # 初始参数向量 [k1, k2, k3, k4, k5]
        # 基于参考实现的初始值
        self.theta0 = np.array([0.0302, 0.0293, -0.0414, -0.0415, -0.0416])
        self.theta = self.theta0.copy()

        # 协方差矩阵
        self.P = initial_covariance * np.eye(5)
        self.P_init = initial_covariance

        # OCV-SOC多项式系数 (5阶多项式，SOC归一化到0-1)
        # 默认使用SP20电池的系数
        if ocv_soc_coeffs is None:
            # 从之前拟合的结果: a5, a4, a3, a2, a1, a0
            self.ocv_coeffs = np.array([
                7.0764914278,    # a5
                -22.6603870282,  # a4
                27.3386349004,   # a3
                -14.5745387772,  # a2
                3.7881828466,    # a1
                3.1958428465     # a0
            ])
        else:
            self.ocv_coeffs = ocv_soc_coeffs

        # 历史状态变量
        self._Uoc = [None, None, None]      # [k, k-1, k-2]
        self._Ut = [None, None, None]       # 端电压
        self._I = [None, None, None]        # 电流

        # 步数计数器
        self.step = 0

        # 辨识结果
        self.R0 = 0.0
        self.R1 = 0.0
        self.R2 = 0.0
        self.C1 = 0.0
        self.C2 = 0.0
        self.tau1 = 0.0
        self.tau2 = 0.0

        # 历史记录
        self.history = {
            'R0': [], 'R1': [], 'R2': [],
            'C1': [], 'C2': [],
            'tau1': [], 'tau2': [],
            'lambda': [],
            'theta': []
        }

    def get_ocv(self, soc: float) -> float:
        """
        根据SOC计算OCV

        Args:
            soc: SOC值 (0-100 或 0-1，自动判断)

        Returns:
            OCV电压值 (V)
        """
        # 归一化SOC到0-1范围
        if soc > 1:
            soc = soc / 100.0

        return np.polyval(self.ocv_coeffs, soc)

    def reset(self):
        """重置辨识器状态"""
        self.theta = self.theta0.copy()
        self.P = self.P_init * np.eye(5)
        self.lamda = self.lambda_init
        self.step = 0
        self._Uoc = [None, None, None]
        self._Ut = [None, None, None]
        self._I = [None, None, None]

        for key in self.history:
            self.history[key] = []

    def _update_history(self, Uoc: float, Ut: float, I: float):
        """更新历史状态变量"""
        # 移位: [k] -> [k-1] -> [k-2]
        self._Uoc[2] = self._Uoc[1]
        self._Uoc[1] = self._Uoc[0]
        self._Uoc[0] = Uoc

        self._Ut[2] = self._Ut[1]
        self._Ut[1] = self._Ut[0]
        self._Ut[0] = Ut

        self._I[2] = self._I[1]
        self._I[1] = self._I[0]
        self._I[0] = I

    def _compute_circuit_params(self) -> Tuple[float, float, float, float, float]:
        """
        从辨识参数theta计算电路参数

        二阶RC模型的传递函数离散化后的系数关系

        Returns:
            (R0, R1, R2, C1, C2)
        """
        k1, k2, k3, k4, k5 = self.theta
        T = self.T

        # 防止除零
        denom = 1 - k4 + k5
        if abs(denom) < 1e-10:
            denom = 1e-10

        # 计算中间变量
        a = (k1 - k2 + k3) / denom
        b = 4 * (k1 - k3) / (T * denom)
        c = 4 * (k1 + k2 + k3) / (T**2 * denom)
        d = 4 * (1 - k5) / (T * denom)
        e = 4 * (1 + k5 + k4) / (T**2 * denom)

        # 计算电路参数
        R0 = a

        # 计算时间常数 (可能有复数解)
        discriminant = d**2 - 4*e
        sqrt_disc = np.sqrt(discriminant + 0j)

        tau1 = (d + sqrt_disc) / (2*e) if abs(e) > 1e-10 else 0.1
        tau2 = (d - sqrt_disc) / (2*e) if abs(e) > 1e-10 else 0.01

        # 计算R1, R2
        tau_diff = tau2 - tau1
        if abs(tau_diff) < 1e-10:
            tau_diff = 1e-10

        R1 = ((b - a*d) - (c - a*e)*tau1) / (e * tau_diff) if abs(e) > 1e-10 else 0.01
        R2 = c/e - a - R1 if abs(e) > 1e-10 else 0.01

        # 计算C1, C2
        C1 = tau1 / R1 if abs(R1) > 1e-10 else 1000
        C2 = tau2 / R2 if abs(R2) > 1e-10 else 1000

        # 取实部并确保物理意义
        R0 = max(np.real(R0), 1e-6)
        R1 = max(np.real(R1), 1e-6)
        R2 = max(np.real(R2), 1e-6)
        C1 = max(np.real(C1), 1e-3)
        C2 = max(np.real(C2), 1e-3)
        tau1 = max(np.real(tau1), 1e-3)
        tau2 = max(np.real(tau2), 1e-3)

        return R0, R1, R2, C1, C2, tau1, tau2

    def identify(
        self,
        voltage: float,
        current: float,
        soc: float,
        step: int = None
    ) -> dict:
        """
        执行一步参数辨识

        Args:
            voltage: 测量端电压 Ut (V)
            current: 测量电流 I (A), 放电为负，充电为正
            soc: 当前SOC估计值 (0-100)
            step: 步数 (可选，如果不提供则自动计数)

        Returns:
            dict: 包含辨识结果的字典
                - R0, R1, R2: 电阻 (Ohm)
                - C1, C2: 电容 (F)
                - tau1, tau2: 时间常数 (s)
                - lambda: 当前遗忘因子
        """
        if step is None:
            self.step += 1
            step = self.step
        else:
            self.step = step

        # 计算OCV
        Uoc = self.get_ocv(soc)

        # 电流符号转换：FFRLS算法内部使用放电为正的约定
        # 输入数据约定：放电为负，充电为正
        # 转换为：放电为正，充电为负
        current_internal = -current

        # 更新历史状态
        self._update_history(Uoc, voltage, current_internal)

        # 前3步只收集数据
        if step < 3:
            result = {
                'R0': self.R0, 'R1': self.R1, 'R2': self.R2,
                'C1': self.C1, 'C2': self.C2,
                'tau1': self.tau1, 'tau2': self.tau2,
                'lambda': self.lamda
            }
            return result

        # 计算电压差 U = Ut - Uoc (极化电压)
        U = self._Ut[0] - self._Uoc[0]
        U_1 = self._Ut[1] - self._Uoc[1] if self._Ut[1] is not None else 0
        U_2 = self._Ut[2] - self._Uoc[2] if self._Ut[2] is not None else 0

        # 检查电流是否足够大以进行有效辨识
        current_magnitude = max(
            abs(self._I[0]) if self._I[0] is not None else 0,
            abs(self._I[1]) if self._I[1] is not None else 0
        )

        # 只有在电流足够大时才更新参数
        if current_magnitude >= self.min_current_threshold:
            # 构造回归向量 phi
            # 基于离散化的二阶RC模型传递函数
            phi = np.array([
                self._I[0],   # I(k)
                self._I[1] if self._I[1] is not None else 0,   # I(k-1)
                self._I[2] if self._I[2] is not None else 0,   # I(k-2)
                U_1,          # U(k-1)
                U_2           # U(k-2)
            ])

            # FFRLS 递推更新
            # 计算增益向量 K
            P_phi = self.P @ phi
            denom = self.lamda + phi.T @ P_phi
            if abs(denom) < 1e-10:
                denom = 1e-10
            K = P_phi / denom

            # 计算预测误差
            y = -U  # 输出 (极化电压的负值)
            y_pred = phi.T @ self.theta
            e = y - y_pred

            # 更新参数向量
            self.theta = self.theta + K * e

            # 更新协方差矩阵
            self.P = (self.P - np.outer(K, phi.T @ self.P)) / self.lamda

            # 自适应遗忘因子更新
            if self.adaptive_lambda:
                phi_K = phi.T @ K
                self.lamda = 1 - (1 - phi_K) * e**2 / self.sigma
                # 限制遗忘因子范围（参考实现）
                if self.lamda <= 0:
                    self.lamda = 0.05

        # 计算电路参数
        R0, R1, R2, C1, C2, tau1, tau2 = self._compute_circuit_params()

        # 更新实例属性
        self.R0, self.R1, self.R2 = R0, R1, R2
        self.C1, self.C2 = C1, C2
        self.tau1, self.tau2 = tau1, tau2

        # 记录历史
        self.history['R0'].append(R0)
        self.history['R1'].append(R1)
        self.history['R2'].append(R2)
        self.history['C1'].append(C1)
        self.history['C2'].append(C2)
        self.history['tau1'].append(tau1)
        self.history['tau2'].append(tau2)
        self.history['lambda'].append(self.lamda)
        self.history['theta'].append(self.theta.copy())

        return {
            'R0': R0, 'R1': R1, 'R2': R2,
            'C1': C1, 'C2': C2,
            'tau1': tau1, 'tau2': tau2,
            'lambda': self.lamda
        }

    def identify_batch(
        self,
        voltage: np.ndarray,
        current: np.ndarray,
        soc: np.ndarray,
        reset_before: bool = True
    ) -> dict:
        """
        批量参数辨识

        Args:
            voltage: 电压序列 (V)
            current: 电流序列 (A)
            soc: SOC序列 (0-100)
            reset_before: 是否在开始前重置辨识器

        Returns:
            dict: 包含所有时刻辨识结果的字典
        """
        if reset_before:
            self.reset()

        n = len(voltage)
        results = {
            'R0': np.zeros(n), 'R1': np.zeros(n), 'R2': np.zeros(n),
            'C1': np.zeros(n), 'C2': np.zeros(n),
            'tau1': np.zeros(n), 'tau2': np.zeros(n),
            'lambda': np.zeros(n)
        }

        for i in range(n):
            result = self.identify(voltage[i], current[i], soc[i])
            for key in results:
                results[key][i] = result[key]

        return results

    def get_model_voltage(
        self,
        current: float,
        soc: float,
        U1: float = None,
        U2: float = None
    ) -> Tuple[float, float, float]:
        """
        使用辨识的参数计算模型电压

        Args:
            current: 电流 (A), 放电为负，充电为正
            soc: SOC (0-100)
            U1: RC1极化电压 (V), 如果为None则从内部状态获取
            U2: RC2极化电压 (V)

        Returns:
            (Ut, U1_new, U2_new): 端电压和更新后的极化电压
        """
        Uoc = self.get_ocv(soc)

        # 电流符号转换：内部使用放电为正的约定
        I = -current

        # 简化计算：假设RC环路已达稳态
        if U1 is None:
            U1 = I * self.R1 * (1 - np.exp(-self.T / self.tau1))
        if U2 is None:
            U2 = I * self.R2 * (1 - np.exp(-self.T / self.tau2))

        # RC环路状态更新 (一阶差分)
        alpha1 = np.exp(-self.T / self.tau1) if self.tau1 > 0 else 0
        alpha2 = np.exp(-self.T / self.tau2) if self.tau2 > 0 else 0

        U1_new = alpha1 * U1 + (1 - alpha1) * I * self.R1
        U2_new = alpha2 * U2 + (1 - alpha2) * I * self.R2

        # 端电压: Ut = Uoc - I*R0 - U1 - U2 (放电时 I>0, Ut<Uoc)
        Ut = Uoc - I * self.R0 - U1_new - U2_new

        return Ut, U1_new, U2_new


def load_ocv_coefficients(csv_path: str = None) -> np.ndarray:
    """
    从CSV文件加载OCV-SOC多项式系数

    Args:
        csv_path: CSV文件路径，如果为None则使用默认路径

    Returns:
        多项式系数数组 (高阶在前)
    """
    if csv_path is None:
        # 默认路径
        default_path = Path(__file__).parent.parent.parent / "dataset" / "OCV-SOC" / "5th_order_polynomial_coefficients.csv"
        csv_path = default_path

    import pandas as pd
    df = pd.read_csv(csv_path)

    # 假设CSV格式: Coefficient, Sample1_SP20-1, Sample2_SP20-3
    # 取第一个样品的系数
    coeffs = df.iloc[:, 1].values  # a0, a1, ..., a5

    # 反转顺序使高阶在前 (numpy.polyval需要)
    return coeffs[::-1]


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
        soc = df['soc_percent'].values
        time = df['time_s'].values

        # 创建辨识器
        ffrls = FFRLS(sample_time=1.0, forgetting_factor=0.998)

        # 批量辨识
        print("Running FFRLS identification...")
        results = ffrls.identify_batch(voltage, current, soc)

        # 绘制结果
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))

        # R0
        axes[0, 0].plot(time[2:], results['R0'][2:], 'b-', linewidth=0.8)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('R0 (Ohm)')
        axes[0, 0].set_title('Ohmic Resistance R0')
        axes[0, 0].grid(True, alpha=0.3)

        # R1, R2
        axes[0, 1].plot(time[2:], results['R1'][2:], 'b-', linewidth=0.8, label='R1')
        axes[0, 1].plot(time[2:], results['R2'][2:], 'r-', linewidth=0.8, label='R2')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('R (Ohm)')
        axes[0, 1].set_title('Polarization Resistances')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # C1, C2
        axes[1, 0].plot(time[2:], results['C1'][2:], 'b-', linewidth=0.8, label='C1')
        axes[1, 0].plot(time[2:], results['C2'][2:], 'r-', linewidth=0.8, label='C2')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('C (F)')
        axes[1, 0].set_title('Polarization Capacitances')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # tau1, tau2
        axes[1, 1].plot(time[2:], results['tau1'][2:], 'b-', linewidth=0.8, label='tau1')
        axes[1, 1].plot(time[2:], results['tau2'][2:], 'r-', linewidth=0.8, label='tau2')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('tau (s)')
        axes[1, 1].set_title('Time Constants')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Lambda
        axes[2, 0].plot(time[2:], results['lambda'][2:], 'g-', linewidth=0.8)
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Lambda')
        axes[2, 0].set_title('Forgetting Factor')
        axes[2, 0].grid(True, alpha=0.3)

        # 电压对比
        axes[2, 1].plot(time, voltage, 'b-', linewidth=0.8, label='Measured')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Voltage (V)')
        axes[2, 1].set_title('Terminal Voltage')
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(Path(__file__).parent.parent.parent / "dataset" / "processed" / "FFRLS_results.png", dpi=150)
        print("Results saved to: dataset/processed/FFRLS_results.png")
        plt.close()  # Close instead of show for non-interactive mode

        # 打印最终参数
        print("\nFinal identified parameters:")
        print(f"  R0 = {results['R0'][-1]*1000:.4f} mOhm")
        print(f"  R1 = {results['R1'][-1]*1000:.4f} mOhm")
        print(f"  R2 = {results['R2'][-1]*1000:.4f} mOhm")
        print(f"  C1 = {results['C1'][-1]:.2f} F")
        print(f"  C2 = {results['C2'][-1]:.2f} F")
        print(f"  tau1 = {results['tau1'][-1]:.4f} s")
        print(f"  tau2 = {results['tau2'][-1]:.4f} s")
    else:
        print(f"Test data not found: {data_path}")
        print("Please run process_battery_data.py first to generate processed data.")
