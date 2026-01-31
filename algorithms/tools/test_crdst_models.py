"""
CRDST 数据集 SOC 估计模型对比测试
Test All SOC Estimation Models on CRDST Dataset

运行方式:
    python algorithms/tools/test_crdst_models.py

输出:
    - 控制台显示各模型性能指标
    - results/graphs/CRDST/ 目录下生成对比图表和CSV文件
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from algorithms.functions.aekf import AEKF
from algorithms.functions.ukf import UKF
from algorithms.functions.sr_ukf import RobustSRUKF
from algorithms.functions.gst_iaekf import GSTIAEKF


# CRDST专用：重写AEKF的update方法，使用激进的Q更新策略
class AEKF_Aggressive(AEKF):
    """CRDST数据集专用AEKF - 使用激进的自适应Q更新策略"""

    def update(self, voltage: float, current: float):
        """
        更新步骤 - 使用激进的Q更新策略（与原始MATLAB CRDST代码一致）
        """
        # 预测端电压
        Ut_pred = self.model.observation(self.x, current)

        # 测量残差（新息）
        y = voltage - Ut_pred

        # 获取观测雅可比矩阵
        C = self.model.get_observation_jacobian(self.x, current)

        # 计算卡尔曼增益
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T / S

        # 状态更新
        self.x = self.x + K * y

        # SOC边界约束
        self.x[0] = np.clip(self.x[0], 0, 1)

        # 协方差更新 - 简化形式（与原始CRDST MATLAB代码一致）
        I_KC = np.eye(self.n_states) - np.outer(K, C)
        self.P = I_KC @ self.P

        # 自适应过程噪声协方差更新 - 激进策略（直接替换，无平滑）
        if self.adaptive_Q:
            self.Q = np.outer(K, K) * (y ** 2)

        return self.x, Ut_pred


class UKF_Aggressive(UKF):
    """CRDST数据集专用UKF - 使用激进的自适应Q更新策略"""

    def update(self, voltage: float, current: float):
        """更新步骤 - UKF激进版本"""
        # 重新生成预测后的Sigma点
        sigma_points = self._generate_sigma_points(self.x, self.P)

        # 通过观测函数传播Sigma点
        n_sigma = sigma_points.shape[1]
        y_points = np.zeros(n_sigma)
        for i in range(n_sigma):
            y_points[i] = self.model.observation(sigma_points[:, i], current)

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
        self.x[0] = np.clip(self.x[0], 0, 1)

        # 协方差更新 - 简化形式
        self.P = self.P - np.outer(K, K) * Pyy

        # 确保对称性
        self.P = (self.P + self.P.T) / 2

        return self.x, y_pred


class SRUKF_Aggressive(RobustSRUKF):
    """CRDST数据集专用SR-UKF - 直接使用父类的鲁棒SR-UKF实现"""
    pass  # SR-UKF已经足够鲁棒，不需要额外修改


class GSTIAEKF_Aggressive(GSTIAEKF):
    """CRDST数据集专用GST-IAEKF - 使用激进的自适应Q更新策略"""

    def update(self, voltage: float, current: float):
        """更新步骤 - GST-IAEKF激进版本，保留门控和强跟踪特性"""
        # 预测端电压
        Ut_pred = self.model.observation(self.x, current)

        # 测量残差
        y = voltage - Ut_pred

        # 观测雅可比矩阵
        C = self.model.get_observation_jacobian(self.x, current)

        # 新息协方差
        S_prior = C @ self.P @ C.T + self.R

        # NIS门控检测
        nis = (y ** 2) / S_prior
        is_gated = False
        if self.enable_nis_gate and nis > self.nis_threshold:
            is_gated = True
            self.R_adaptive = self.R * self.nis_R_scale
        else:
            self.R_adaptive = self.R

        # 计算卡尔曼增益
        S = C @ self.P @ C.T + self.R_adaptive
        K = self.P @ C.T / S

        # 状态更新
        self.x = self.x + K * y
        self.x[0] = np.clip(self.x[0], 0, 1)

        # 协方差更新 - 简化形式（激进策略）
        I_KC = np.eye(self.n) - np.outer(K, C)
        self.P = I_KC @ self.P

        # 激进的Q更新（直接替换）
        if self.enable_qr_adaptive:
            self.Q = np.outer(K, K) * (y ** 2)

        # 强跟踪因子
        lambda_st = 1.0
        if self.enable_strong_tracking:
            expected_innovation = np.sqrt(S_prior)
            actual_innovation = abs(y)
            if actual_innovation > 2 * expected_innovation:
                lambda_st = min(self.lambda_max, actual_innovation / expected_innovation)

        return self.x, Ut_pred, lambda_st, is_gated


# ============================================================================
#                           用户配置区域 (Configuration)
# ============================================================================

# 选择要运行的模型 (可选: "AEKF", "UKF", "SR-UKF", "GST-IAEKF")
# 设置为 None 或 [] 运行所有模型
RUN_MODELS = None

# SOC过滤范围 (%)
SOC_MIN = 10.0
SOC_MAX = 100.0

# 初始 SOC 偏移量（相对于真实初始SOC）
# 例如: -0.1 表示初始估计比真实值低 10%
# 注意：CRDST数据集在高SOC区域OCV曲线很平坦，建议使用较小的偏移量
INITIAL_SOC_OFFSET = 0.0  # 改为0尝试，先测试模型本身的性能

# CRDST 数据集路径
CRDST_DATA_PATH = PROJECT_ROOT / "dataset" / "processed" / "CRDST" / "CRDST_96SOC.csv"
CRDST_OCV_PATH = PROJECT_ROOT / "dataset" / "processed" / "CRDST" / "OCV-SOC" / "Sample1" / "Discharge_OCV_SOC_data.csv"

# 结果输出目录
RESULTS_DIR = PROJECT_ROOT / "results" / "graphs" / "CRDST"

# 温度设置（CRDST 数据集使用专用的OCV-SOC曲线）
TEMPERATURE = "CRDST"  # 使用CRDST专用的7阶多项式OCV系数

# ============================================================================
#                           配置区域结束
# ============================================================================

ALL_MODELS = ["AEKF", "GST-IAEKF"]
# , "UKF", "SR-UKF"

def load_crdst_data(data_path: Path, soc_min: float, soc_max: float):
    """加载并过滤 CRDST 数据"""
    df = pd.read_csv(data_path)

    print(f"\nLoading CRDST dataset...")
    print(f"  Total samples: {len(df)}")
    print(f"  SOC range (raw): {df['soc_percent'].min():.2f}% - {df['soc_percent'].max():.2f}%")
    print(f"  Duration: {df['time_s'].max() / 60:.1f} minutes")

    # 过滤 SOC 范围
    mask = (df['soc_percent'] >= soc_min) & (df['soc_percent'] <= soc_max)
    df_filtered = df[mask].reset_index(drop=True)

    print(f"  Filtered SOC range: {soc_min}% - {soc_max}%")
    print(f"  Filtered samples: {len(df_filtered)}")
    print(f"  Filtered duration: {df_filtered['time_s'].max() / 60:.1f} minutes")

    # CRDST数据符号转换：
    # 原始数据"放电为正"(0.206A) -> 转换为"放电为负"(-0.206A)
    # 符合CALCE约定和battery_model/FFRLS的接口要求
    return {
        'voltage': df_filtered['voltage_V'].values,
        'current': -df_filtered['current_A'].values,  # 放电正->放电负
        'soc_true': df_filtered['soc_percent'].values,
        'time': df_filtered['time_s'].values
    }


def calculate_metrics(soc_estimated: np.ndarray, soc_true: np.ndarray):
    """计算误差指标"""
    error = soc_estimated - soc_true
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))
    std_error = np.std(error)
    return {
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'std_error': std_error,
        'error': error
    }


def run_aekf(data: dict, temperature: str, initial_soc_offset: float):
    """运行AEKF算法"""
    initial_soc = data['soc_true'][0] / 100 + initial_soc_offset
    initial_soc = np.clip(initial_soc, 0.01, 0.99)

    # CRDST数据集使用原始成功代码的参数配置
    # 关键：Q矩阵要大，才能快速跟踪SOC变化
    process_noise = np.array([
        [4e-2, 0, 0],      # SOC过程噪声 (0.04，比默认值大40000倍)
        [0, 2e-4, 0],      # U1过程噪声
        [0, 0, 2e-4]       # U2过程噪声
    ])

    initial_covariance = np.array([
        [4e-5, 0, 0],      # SOC初始协方差
        [0, 2e-4, 0],      # U1初始协方差
        [0, 0, 2e-4]       # U2初始协方差
    ])

    # 使用CRDST专用的激进AEKF（与原始MATLAB代码一致）
    aekf = AEKF_Aggressive(
        initial_soc=initial_soc,
        capacity_Ah=230*3600/112/3600,  # 2.054 Ah
        sample_time=1.0,
        process_noise=process_noise,
        measurement_noise=2e-4,
        initial_covariance=initial_covariance,
        use_online_param_id=True,
        adaptive_Q=True,  # CRDST: 使用激进的自适应Q（直接替换）
        temperature=temperature
    )

    results = aekf.estimate_batch(
        data['voltage'], data['current'], data['soc_true'], initial_soc
    )

    soc_est = results['SOC_percent']
    metrics = calculate_metrics(soc_est, data['soc_true'])
    return soc_est, metrics


def run_ukf(data: dict, temperature: str, initial_soc_offset: float):
    """运行UKF算法"""
    initial_soc = data['soc_true'][0] / 100 + initial_soc_offset
    initial_soc = np.clip(initial_soc, 0.01, 0.99)

    # CRDST数据集使用较大的过程噪声协方差
    process_noise = np.array([
        [4e-2, 0, 0],      # SOC过程噪声
        [0, 2e-4, 0],      # U1过程噪声
        [0, 0, 2e-4]       # U2过程噪声
    ])

    initial_covariance = np.array([
        [4e-5, 0, 0],
        [0, 2e-4, 0],
        [0, 0, 2e-4]
    ])

    # 使用CRDST专用的激进UKF
    ukf = UKF_Aggressive(
        initial_soc=initial_soc,
        capacity_Ah=230*3600/112/3600,  # 2.054 Ah
        sample_time=1.0,
        process_noise=process_noise,
        measurement_noise=2e-4,
        initial_covariance=initial_covariance,
        use_online_param_id=True,
        temperature=temperature
    )

    results = ukf.estimate_batch(
        data['voltage'], data['current'], data['soc_true'], initial_soc
    )

    soc_est = results['SOC_percent']
    metrics = calculate_metrics(soc_est, data['soc_true'])
    return soc_est, metrics


def run_sr_ukf(data: dict, temperature: str, initial_soc_offset: float):
    """运行SR-UKF算法"""
    initial_soc = data['soc_true'][0] / 100 + initial_soc_offset
    initial_soc = np.clip(initial_soc, 0.01, 0.99)

    # CRDST数据集使用较大的过程噪声协方差
    process_noise = np.array([
        [4e-2, 0, 0],
        [0, 2e-4, 0],
        [0, 0, 2e-4]
    ])

    initial_covariance = np.array([
        [4e-5, 0, 0],
        [0, 2e-4, 0],
        [0, 0, 2e-4]
    ])

    # 使用CRDST专用的激进SR-UKF
    sr_ukf = SRUKF_Aggressive(
        initial_soc=initial_soc,
        capacity_Ah=230*3600/112/3600,  # 2.054 Ah
        sample_time=1.0,
        process_noise=process_noise,
        measurement_noise=2e-4,
        initial_covariance=initial_covariance,
        use_online_param_id=True,
        enable_nis_gate=True,
        enable_student_t=True,
        adaptive_nu=True,
        temperature=temperature
    )

    results = sr_ukf.estimate_batch(
        data['voltage'], data['current'], data['soc_true'], initial_soc
    )

    soc_est = results['SOC_percent']
    metrics = calculate_metrics(soc_est, data['soc_true'])
    return soc_est, metrics


def run_gst_iaekf(data: dict, temperature: str, initial_soc_offset: float):
    """运行GST-IAEKF算法"""
    initial_soc = data['soc_true'][0] / 100 + initial_soc_offset
    initial_soc = np.clip(initial_soc, 0.01, 0.99)

    # CRDST数据集使用较大的过程噪声协方差
    process_noise = np.array([
        [4e-2, 0, 0],
        [0, 2e-4, 0],
        [0, 0, 2e-4]
    ])

    initial_covariance = np.array([
        [4e-5, 0, 0],
        [0, 2e-4, 0],
        [0, 0, 2e-4]
    ])

    # 使用CRDST专用的激进GST-IAEKF
    gst_iaekf = GSTIAEKF_Aggressive(
        initial_soc=initial_soc,
        capacity_Ah=230*3600/112/3600,  # 2.054 Ah
        sample_time=1.0,
        process_noise=process_noise,
        measurement_noise=2e-4,
        initial_covariance=initial_covariance,
        use_online_param_id=True,
        enable_nis_gate=True,
        enable_strong_tracking=True,
        enable_qr_adaptive=True,  # CRDST: 启用激进的Q/R自适应
        lambda_max=3.0,
        rho=0.95,
        temperature=temperature
    )

    results = gst_iaekf.estimate_batch(
        data['voltage'], data['current'], data['soc_true'], initial_soc
    )

    soc_est = results['SOC_percent']
    metrics = calculate_metrics(soc_est, data['soc_true'])
    return soc_est, metrics


def run_model(model_name: str, data: dict, temperature: str, initial_soc_offset: float):
    """运行指定模型，返回 (soc_estimated, metrics)"""
    runners = {
        'AEKF': run_aekf,
        'UKF': run_ukf,
        'SR-UKF': run_sr_ukf,
        'GST-IAEKF': run_gst_iaekf
    }
    return runners[model_name](data, temperature, initial_soc_offset)


def plot_soc_comparison(data: dict, model_results: dict, save_dir: Path):
    """绘制 SOC 对比图和误差图"""
    time = data['time'] / 60  # 转换为分钟
    soc_true = data['soc_true']

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    colors = {'AEKF': '#1f77b4', 'UKF': '#ff7f0e', 'SR-UKF': '#2ca02c', 'GST-IAEKF': '#d62728'}
    linestyles = {'AEKF': '--', 'UKF': '-.', 'SR-UKF': ':', 'GST-IAEKF': '-'}

    # 上图: SOC对比
    axes[0].plot(time, soc_true, 'k-', linewidth=2.5, label='True SOC', alpha=0.9, zorder=5)
    for model_name, (soc_est, metrics) in model_results.items():
        axes[0].plot(time, soc_est,
                     color=colors.get(model_name, 'gray'),
                     linestyle=linestyles.get(model_name, '-'),
                     linewidth=1.8,
                     alpha=0.85,
                     label=f'{model_name} (RMSE={metrics["rmse"]:.3f}%)')

    axes[0].set_xlabel('Time (min)', fontsize=12)
    axes[0].set_ylabel('SOC (%)', fontsize=12)
    axes[0].set_title('CRDST Dataset - SOC Estimation Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10, framealpha=0.9)
    axes[0].grid(True, alpha=0.3)

    # 下图: 误差对比
    for model_name, (soc_est, metrics) in model_results.items():
        axes[1].plot(time, metrics['error'],
                     color=colors.get(model_name, 'gray'),
                     linewidth=1.2, alpha=0.8,
                     label=f'{model_name} (Max={metrics["max_error"]:.3f}%, MAE={metrics["mae"]:.3f}%)')

    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1.0, alpha=0.5)
    axes[1].set_xlabel('Time (min)', fontsize=12)
    axes[1].set_ylabel('SOC Error (%)', fontsize=12)
    axes[1].set_title('Estimation Error Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=10, framealpha=0.9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "CRDST_soc_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSOC comparison plot saved to: {save_path}")


def plot_metrics_comparison(model_results: dict, save_dir: Path):
    """绘制各模型指标对比柱状图"""
    models = list(model_results.keys())
    rmse_values = [metrics['rmse'] for _, metrics in model_results.values()]
    mae_values = [metrics['mae'] for _, metrics in model_results.values()]
    max_error_values = [metrics['max_error'] for _, metrics in model_results.values()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = {'AEKF': '#1f77b4', 'UKF': '#ff7f0e', 'SR-UKF': '#2ca02c', 'GST-IAEKF': '#d62728'}
    bar_colors = [colors.get(model, 'gray') for model in models]

    # RMSE
    bars1 = axes[0].bar(models, rmse_values, color=bar_colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('RMSE (%)', fontsize=12)
    axes[0].set_title('Root Mean Square Error', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars1, rmse_values)):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # MAE
    bars2 = axes[1].bar(models, mae_values, color=bar_colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('MAE (%)', fontsize=12)
    axes[1].set_title('Mean Absolute Error', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars2, mae_values)):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Max Error
    bars3 = axes[2].bar(models, max_error_values, color=bar_colors, alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('Max Error (%)', fontsize=12)
    axes[2].set_title('Maximum Absolute Error', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars3, max_error_values)):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('CRDST Dataset - Model Performance Metrics', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = save_dir / "CRDST_metrics_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Metrics comparison plot saved to: {save_path}")


def plot_error_distribution(model_results: dict, save_dir: Path):
    """绘制误差分布直方图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = {'AEKF': '#1f77b4', 'UKF': '#ff7f0e', 'SR-UKF': '#2ca02c', 'GST-IAEKF': '#d62728'}

    for idx, (model_name, (soc_est, metrics)) in enumerate(model_results.items()):
        error = metrics['error']

        # 绘制直方图
        axes[idx].hist(error, bins=50, color=colors.get(model_name, 'gray'),
                       alpha=0.7, edgecolor='black', density=True)

        # 叠加高斯曲线
        mu, sigma = np.mean(error), np.std(error)
        x = np.linspace(error.min(), error.max(), 100)
        axes[idx].plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu)/sigma)**2),
                       'r-', linewidth=2, label=f'Gaussian (μ={mu:.3f}, σ={sigma:.3f})')

        axes[idx].axvline(x=0, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
        axes[idx].set_xlabel('Error (%)', fontsize=11)
        axes[idx].set_ylabel('Density', fontsize=11)
        axes[idx].set_title(f'{model_name} Error Distribution', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('CRDST Dataset - Error Distribution Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()

    save_path = save_dir / "CRDST_error_distribution.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Error distribution plot saved to: {save_path}")


def save_results_csv(model_results: dict, save_dir: Path):
    """保存结果到 CSV 文件"""
    # 创建汇总数据
    summary_data = []
    for model_name, (soc_est, metrics) in model_results.items():
        summary_data.append({
            'Model': model_name,
            'RMSE (%)': metrics['rmse'],
            'MAE (%)': metrics['mae'],
            'Max Error (%)': metrics['max_error'],
            'Std Error (%)': metrics['std_error']
        })

    df_summary = pd.DataFrame(summary_data)

    # 按 RMSE 排序
    df_summary = df_summary.sort_values('RMSE (%)')

    # 保存汇总结果
    summary_path = save_dir / "CRDST_results_summary.csv"
    df_summary.to_csv(summary_path, index=False, float_format='%.6f')

    print(f"\nResults summary saved to: {summary_path}")

    return df_summary


def print_summary_table(df_summary: pd.DataFrame):
    """打印汇总表格到控制台"""
    print("\n" + "=" * 80)
    print("CRDST DATASET - MODEL PERFORMANCE SUMMARY")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)


def main():
    """主函数"""
    print("=" * 80)
    print("CRDST Dataset - SOC Estimation Model Comparison")
    print("=" * 80)

    # 创建结果目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 检查数据文件是否存在
    if not CRDST_DATA_PATH.exists():
        print(f"\nError: CRDST data file not found at {CRDST_DATA_PATH}")
        print("Please ensure the CRDST dataset is processed and placed correctly.")
        return

    # 加载数据
    data = load_crdst_data(CRDST_DATA_PATH, SOC_MIN, SOC_MAX)

    # 确定要运行的模型
    models_to_run = RUN_MODELS if RUN_MODELS else ALL_MODELS

    print(f"\nRunning models: {', '.join(models_to_run)}")
    print(f"Dataset: {TEMPERATURE} (25°C)")
    print(f"Initial SOC offset: {INITIAL_SOC_OFFSET*100:.1f}%")
    print(f"True initial SOC: {data['soc_true'][0]:.2f}%")
    print(f"Estimated initial SOC: {(data['soc_true'][0]/100 + INITIAL_SOC_OFFSET)*100:.2f}%")

    # 运行所有模型
    model_results = {}
    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Running {model_name}...")
        print(f"{'='*60}")

        try:
            soc_est, metrics = run_model(model_name, data, TEMPERATURE, INITIAL_SOC_OFFSET)
            model_results[model_name] = (soc_est, metrics)

            print(f"  RMSE: {metrics['rmse']:.4f}%")
            print(f"  MAE:  {metrics['mae']:.4f}%")
            print(f"  Max Error: {metrics['max_error']:.4f}%")
            print(f"  Std Error: {metrics['std_error']:.4f}%")
            print(f"  {model_name} completed successfully!")

        except Exception as e:
            print(f"  Error running {model_name}: {e}")
            import traceback
            traceback.print_exc()

    if not model_results:
        print("\nNo models ran successfully. Exiting.")
        return

    # 保存结果到 CSV
    df_summary = save_results_csv(model_results, RESULTS_DIR)

    # 打印汇总表格
    print_summary_table(df_summary)

    # 生成对比图表
    print("\nGenerating comparison plots...")
    plot_soc_comparison(data, model_results, RESULTS_DIR)
    plot_metrics_comparison(model_results, RESULTS_DIR)
    plot_error_distribution(model_results, RESULTS_DIR)

    print("\n" + "=" * 80)
    print("All tasks completed successfully!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
