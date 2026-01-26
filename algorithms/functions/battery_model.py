"""
Second-Order RC (Thevenin) Battery Model
二阶RC等效电路电池模型

Model Structure:
    Uoc ----[ R0 ]----+----[ R1 ]----+----[ R2 ]----+---- Ut
                      |              |              |
                     [C1]          [C2]            |
                      |              |              |
                      +--------------+--------------+

State Variables:
    x = [SOC, U1, U2]^T

State Equations (Discrete):
    SOC(k+1) = SOC(k) - (dt / Qn) * I(k)
    U1(k+1) = exp(-dt/tau1) * U1(k) + R1 * (1 - exp(-dt/tau1)) * I(k)
    U2(k+1) = exp(-dt/tau2) * U2(k) + R2 * (1 - exp(-dt/tau2)) * I(k)

Observation Equation:
    Ut = OCV(SOC) - R0 * I - U1 - U2

Current Convention:
    - Input: Discharge negative, Charge positive (CALCE convention)
    - Internal: Discharge positive, Charge negative (Model convention)

Author: Auto-generated for CALCE dataset
Date: 2024
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class BatteryModel2RC:
    """
    二阶RC等效电路电池模型

    用于EKF/UKF等状态估计算法的电池模型。
    """

    def __init__(
        self,
        capacity_Ah: float = 2.0,
        sample_time: float = 1.0,
        R0: float = 0.07,
        R1: float = 0.02,
        R2: float = 0.01,
        C1: float = 1000.0,
        C2: float = 100.0,
        ocv_coeffs: np.ndarray = None
    ):
        """
        初始化电池模型

        Args:
            capacity_Ah: 电池额定容量 (Ah)
            sample_time: 采样时间 (s)
            R0: 欧姆内阻 (Ohm)
            R1: 第一RC环电阻 (Ohm)
            R2: 第二RC环电阻 (Ohm)
            C1: 第一RC环电容 (F)
            C2: 第二RC环电容 (F)
            ocv_coeffs: OCV-SOC多项式系数 (高阶在前，SOC归一化到0-1)
        """
        # 电池容量 (A·s)
        self.Qn = capacity_Ah * 3600
        self.capacity_Ah = capacity_Ah

        # 采样时间
        self.dt = sample_time

        # 电路参数
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.C1 = C1
        self.C2 = C2

        # 时间常数
        self.tau1 = R1 * C1
        self.tau2 = R2 * C2

        # OCV-SOC多项式系数 (5阶，基于Sample1放电数据)
        if ocv_coeffs is None:
            # 默认使用Sample1的系数
            self.ocv_coeffs = np.array([
                7.0764914278,    # a5
                -22.6603870282,  # a4
                27.3386349004,   # a3
                -14.5745387772,  # a2
                3.7881828466,    # a1
                3.1958428465     # a0
            ])
        else:
            self.ocv_coeffs = ocv_coeffs

        # OCV导数系数 (用于EKF的雅可比矩阵)
        self.ocv_deriv_coeffs = np.polyder(self.ocv_coeffs)

        # 状态维度
        self.n_states = 3

    def update_parameters(self, R0: float, R1: float, R2: float, C1: float, C2: float):
        """
        更新电路参数（用于在线参数辨识）

        Args:
            R0, R1, R2: 电阻 (Ohm)
            C1, C2: 电容 (F)
        """
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.C1 = C1
        self.C2 = C2
        self.tau1 = R1 * C1
        self.tau2 = R2 * C2

    def get_ocv(self, soc: float) -> float:
        """
        根据SOC计算OCV

        Args:
            soc: SOC值 (0-1 或 0-100，自动判断)

        Returns:
            OCV电压 (V)
        """
        # 归一化SOC到0-1
        if soc > 1:
            soc = soc / 100.0
        soc = np.clip(soc, 0, 1)
        return np.polyval(self.ocv_coeffs, soc)

    def get_ocv_derivative(self, soc: float) -> float:
        """
        计算dOCV/dSOC (用于EKF雅可比矩阵)

        Args:
            soc: SOC值 (0-1 或 0-100)

        Returns:
            dOCV/dSOC (V/1)，注意SOC是归一化的
        """
        if soc > 1:
            soc = soc / 100.0
        soc = np.clip(soc, 0, 1)
        return np.polyval(self.ocv_deriv_coeffs, soc)

    def state_transition(self, x: np.ndarray, I: float) -> np.ndarray:
        """
        状态转移函数 x(k+1) = f(x(k), I(k))

        Args:
            x: 状态向量 [SOC, U1, U2]，SOC为0-1
            I: 电流 (A)，放电为负，充电为正（CALCE约定）

        Returns:
            x_next: 下一时刻状态向量
        """
        SOC, U1, U2 = x

        # 电流符号转换：放电为负 -> 放电为正
        I_internal = -I

        # 离散化参数
        a1 = np.exp(-self.dt / self.tau1) if self.tau1 > 0 else 0
        a2 = np.exp(-self.dt / self.tau2) if self.tau2 > 0 else 0
        b1 = self.R1 * (1 - a1)
        b2 = self.R2 * (1 - a2)

        # 状态更新
        SOC_next = SOC - (self.dt / self.Qn) * I_internal
        U1_next = a1 * U1 + b1 * I_internal
        U2_next = a2 * U2 + b2 * I_internal

        # SOC边界约束
        SOC_next = np.clip(SOC_next, 0, 1)

        return np.array([SOC_next, U1_next, U2_next])

    def observation(self, x: np.ndarray, I: float) -> float:
        """
        观测函数 y = h(x, I)

        Args:
            x: 状态向量 [SOC, U1, U2]
            I: 电流 (A)，放电为负，充电为正

        Returns:
            Ut: 端电压 (V)
        """
        SOC, U1, U2 = x

        # 电流符号转换
        I_internal = -I

        # 计算OCV
        OCV = self.get_ocv(SOC)

        # 端电压
        Ut = OCV - self.R0 * I_internal - U1 - U2

        return Ut

    def get_state_jacobian(self, x: np.ndarray, I: float) -> np.ndarray:
        """
        状态转移雅可比矩阵 A = df/dx

        Args:
            x: 状态向量
            I: 电流

        Returns:
            A: 3x3 雅可比矩阵
        """
        a1 = np.exp(-self.dt / self.tau1) if self.tau1 > 0 else 0
        a2 = np.exp(-self.dt / self.tau2) if self.tau2 > 0 else 0

        A = np.array([
            [1, 0, 0],
            [0, a1, 0],
            [0, 0, a2]
        ])

        return A

    def get_observation_jacobian(self, x: np.ndarray, I: float) -> np.ndarray:
        """
        观测雅可比矩阵 C = dh/dx

        Args:
            x: 状态向量
            I: 电流

        Returns:
            C: 1x3 雅可比矩阵（作为行向量）
        """
        SOC = x[0]

        # dOCV/dSOC
        dOCV_dSOC = self.get_ocv_derivative(SOC)

        # C = [dUt/dSOC, dUt/dU1, dUt/dU2] = [dOCV/dSOC, -1, -1]
        C = np.array([dOCV_dSOC, -1, -1])

        return C

    def get_input_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        输入矩阵 B (状态方程中的控制输入系数)

        x(k+1) = A*x(k) + B*I(k)

        Args:
            x: 状态向量

        Returns:
            B: 3x1 输入矩阵（作为列向量）
        """
        a1 = np.exp(-self.dt / self.tau1) if self.tau1 > 0 else 0
        a2 = np.exp(-self.dt / self.tau2) if self.tau2 > 0 else 0
        b1 = self.R1 * (1 - a1)
        b2 = self.R2 * (1 - a2)

        # 注意：这里的B对应内部电流（放电为正）
        # 在使用时需要对输入电流取反
        B = np.array([
            -self.dt / self.Qn,  # SOC: -dt/Qn * I (放电时I>0, SOC减小)
            b1,                   # U1
            b2                    # U2
        ])

        return B


def load_battery_model(
    ffrls_results: dict = None,
    capacity_Ah: float = 2.0,
    sample_time: float = 1.0
) -> BatteryModel2RC:
    """
    创建电池模型，可选择使用FFRLS辨识的参数

    Args:
        ffrls_results: FFRLS辨识结果字典，包含R0, R1, R2, C1, C2
        capacity_Ah: 电池容量
        sample_time: 采样时间

    Returns:
        BatteryModel2RC实例
    """
    if ffrls_results is not None:
        return BatteryModel2RC(
            capacity_Ah=capacity_Ah,
            sample_time=sample_time,
            R0=ffrls_results.get('R0', 0.07),
            R1=ffrls_results.get('R1', 0.02),
            R2=ffrls_results.get('R2', 0.01),
            C1=ffrls_results.get('C1', 1000),
            C2=ffrls_results.get('C2', 100)
        )
    else:
        # 使用默认参数
        return BatteryModel2RC(
            capacity_Ah=capacity_Ah,
            sample_time=sample_time
        )


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = BatteryModel2RC()

    print("Battery Model Test")
    print("=" * 40)
    print(f"Capacity: {model.capacity_Ah} Ah")
    print(f"Sample time: {model.dt} s")
    print(f"R0: {model.R0*1000:.2f} mOhm")
    print(f"R1: {model.R1*1000:.2f} mOhm")
    print(f"R2: {model.R2*1000:.2f} mOhm")
    print(f"tau1: {model.tau1:.2f} s")
    print(f"tau2: {model.tau2:.2f} s")

    # 测试OCV计算
    print("\nOCV-SOC:")
    for soc in [0.1, 0.5, 0.8, 1.0]:
        print(f"  SOC={soc*100:.0f}%: OCV={model.get_ocv(soc):.4f}V")

    # 测试状态转移
    print("\nState Transition Test:")
    x = np.array([0.8, 0, 0])  # 初始状态
    I = -1.0  # 1A放电
    print(f"  Initial: SOC={x[0]*100:.1f}%, U1={x[1]:.4f}V, U2={x[2]:.4f}V")

    for k in range(5):
        x = model.state_transition(x, I)
        Ut = model.observation(x, I)
        print(f"  Step {k+1}: SOC={x[0]*100:.2f}%, U1={x[1]*1000:.2f}mV, U2={x[2]*1000:.2f}mV, Ut={Ut:.4f}V")
