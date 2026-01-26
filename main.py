"""
Main script for SOC Estimation Algorithms Comparison
SOC估计算法主运行脚本

运行方式:
    python main.py

输出:
    - 控制台显示各算法性能指标
    - dataset/processed/ 目录下生成结果图
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from algorithms.functions.aekf import AEKF
from algorithms.functions.ukf import UKF
from algorithms.functions.sr_ukf import RobustSRUKF
from algorithms.functions.gst_iaekf import GSTIAEKF


def load_data(data_path: Path, soc_min: float = 10.0, soc_max: float = 100.0):
    """
    加载并过滤数据

    Args:
        data_path: 数据文件路径
        soc_min: 最小SOC阈值 (%)
        soc_max: 最大SOC阈值 (%)

    Returns:
        过滤后的数据字典
    """
    df = pd.read_csv(data_path)

    # 过滤SOC范围
    mask = (df['soc_percent'] >= soc_min) & (df['soc_percent'] <= soc_max)
    df_filtered = df[mask].reset_index(drop=True)

    print(f"原始数据点: {len(df)}")
    print(f"过滤后数据点: {len(df_filtered)} (SOC: {soc_min}% ~ {soc_max}%)")

    return {
        'voltage': df_filtered['voltage_V'].values,
        'current': df_filtered['current_A'].values,
        'soc_true': df_filtered['soc_percent'].values,
        'time': df_filtered['time_s'].values
    }


def calculate_metrics(soc_estimated: np.ndarray, soc_true: np.ndarray):
    """计算误差指标"""
    error = soc_estimated - soc_true
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))
    return {'rmse': rmse, 'mae': mae, 'max_error': max_error, 'error': error}


def run_aekf(data: dict):
    """运行AEKF算法"""
    print("\n" + "="*50)
    print("Running AEKF...")
    print("="*50)

    initial_soc = data['soc_true'][0] / 100
    aekf = AEKF(
        initial_soc=initial_soc,
        capacity_Ah=2.0,
        sample_time=1.0,
        use_online_param_id=True,
        adaptive_Q=True
    )

    results = aekf.estimate_batch(
        data['voltage'], data['current'], data['soc_true'], initial_soc
    )

    metrics = calculate_metrics(results['SOC_percent'], data['soc_true'])
    print(f"  RMSE: {metrics['rmse']:.4f}%")
    print(f"  MAE: {metrics['mae']:.4f}%")
    print(f"  Max Error: {metrics['max_error']:.4f}%")

    return results, metrics


def run_ukf(data: dict):
    """运行UKF算法"""
    print("\n" + "="*50)
    print("Running UKF...")
    print("="*50)

    initial_soc = data['soc_true'][0] / 100
    ukf = UKF(
        initial_soc=initial_soc,
        capacity_Ah=2.0,
        sample_time=1.0,
        use_online_param_id=True
    )

    results = ukf.estimate_batch(
        data['voltage'], data['current'], data['soc_true'], initial_soc
    )

    metrics = calculate_metrics(results['SOC_percent'], data['soc_true'])
    print(f"  RMSE: {metrics['rmse']:.4f}%")
    print(f"  MAE: {metrics['mae']:.4f}%")
    print(f"  Max Error: {metrics['max_error']:.4f}%")

    return results, metrics


def run_sr_ukf(data: dict):
    """运行Robust SR-UKF算法"""
    print("\n" + "="*50)
    print("Running Robust SR-UKF...")
    print("="*50)

    initial_soc = data['soc_true'][0] / 100
    sr_ukf = RobustSRUKF(
        initial_soc=initial_soc,
        capacity_Ah=2.0,
        sample_time=1.0,
        use_online_param_id=True,
        enable_nis_gate=True,
        enable_student_t=True,
        adaptive_nu=True
    )

    results = sr_ukf.estimate_batch(
        data['voltage'], data['current'], data['soc_true'], initial_soc
    )

    metrics = calculate_metrics(results['SOC_percent'], data['soc_true'])
    nis_triggered = np.sum(results['NIS'] > 6.63)

    print(f"  RMSE: {metrics['rmse']:.4f}%")
    print(f"  MAE: {metrics['mae']:.4f}%")
    print(f"  Max Error: {metrics['max_error']:.4f}%")
    print(f"  NIS Gate Triggered: {nis_triggered} times")

    metrics['nis_triggered'] = nis_triggered
    return results, metrics


def run_gst_iaekf(data: dict):
    """运行GST-IAEKF算法"""
    print("\n" + "="*50)
    print("Running GST-IAEKF...")
    print("="*50)

    initial_soc = data['soc_true'][0] / 100
    gst_iaekf = GSTIAEKF(
        initial_soc=initial_soc,
        capacity_Ah=2.0,
        sample_time=1.0,
        use_online_param_id=True,
        enable_nis_gate=True,
        enable_strong_tracking=True,
        enable_qr_adaptive=False,  # 禁用Q/R自适应（干净数据不需要）
        lambda_max=3.0,            # 适度的强跟踪因子
        rho=0.95
    )

    results = gst_iaekf.estimate_batch(
        data['voltage'], data['current'], data['soc_true'], initial_soc
    )

    metrics = calculate_metrics(results['SOC_percent'], data['soc_true'])
    gate_triggered = np.sum(results['gate_triggered'])

    print(f"  RMSE: {metrics['rmse']:.4f}%")
    print(f"  MAE: {metrics['mae']:.4f}%")
    print(f"  Max Error: {metrics['max_error']:.4f}%")
    print(f"  Gate Triggered: {gate_triggered} times")

    metrics['gate_triggered'] = gate_triggered
    return results, metrics


def plot_comparison(data: dict, results_dict: dict, save_path: Path):
    """绘制对比图"""
    time = data['time']
    soc_true = data['soc_true']

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # SOC对比图（全局视图）
    axes[0].plot(time, soc_true, 'b-', linewidth=2, label='True SOC')

    colors = {'AEKF': 'r', 'UKF': 'g', 'SR-UKF': 'm', 'GST-IAEKF': 'c'}
    linestyles = {'AEKF': '--', 'UKF': '-.', 'SR-UKF': ':', 'GST-IAEKF': '-'}

    for name, (results, metrics) in results_dict.items():
        axes[0].plot(
            time, results['SOC_percent'],
            color=colors[name], linestyle=linestyles[name], linewidth=1.5,
            label=f'{name} (RMSE={metrics["rmse"]:.3f}%)'
        )

    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('SOC (%)', fontsize=12)
    axes[0].set_title('SOC Estimation Comparison (80% → 10%)', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([5, 85])

    # 误差对比图（关键差异所在）
    for name, (results, metrics) in results_dict.items():
        axes[1].plot(
            time, metrics['error'],
            color=colors[name], linewidth=1.0, alpha=0.9,
            label=f'{name} (Max={metrics["max_error"]:.2f}%)'
        )

    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('SOC Error (%)', fontsize=12)
    axes[1].set_title('Estimation Error Comparison (Key Differences Here!)', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # 误差绝对值累积分布（更直观展示差异）
    for name, (results, metrics) in results_dict.items():
        abs_error_sorted = np.sort(np.abs(metrics['error']))
        cdf = np.arange(1, len(abs_error_sorted) + 1) / len(abs_error_sorted) * 100
        axes[2].plot(abs_error_sorted, cdf, color=colors[name], linewidth=2,
                     label=f'{name} (95%: {np.percentile(np.abs(metrics["error"]), 95):.2f}%)')

    axes[2].set_xlabel('Absolute SOC Error (%)', fontsize=12)
    axes[2].set_ylabel('Cumulative Percentage (%)', fontsize=12)
    axes[2].set_title('Error CDF (Cumulative Distribution - Left is Better)', fontsize=14)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, max(3.5, max(m['max_error'] for _, m in results_dict.values()) * 1.1)])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nComparison plot saved to: {save_path}")
    plt.close()


def plot_sr_ukf_details(data: dict, results: dict, metrics: dict, save_path: Path):
    """绘制SR-UKF详细结果"""
    time = data['time']
    soc_true = data['soc_true']

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # SOC估计
    axes[0].plot(time, soc_true, 'b-', linewidth=1.5, label='True SOC')
    axes[0].plot(time, results['SOC_percent'], 'r--', linewidth=1.5, label='SR-UKF Estimated')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('SOC (%)')
    axes[0].set_title(f'Robust SR-UKF SOC Estimation (RMSE={metrics["rmse"]:.4f}%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 误差
    axes[1].plot(time, metrics['error'], 'g-', linewidth=0.8)
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('SOC Error (%)')
    axes[1].set_title('Estimation Error')
    axes[1].grid(True, alpha=0.3)

    # NIS和鲁棒权重
    ax2_twin = axes[2].twinx()
    axes[2].plot(time, results['NIS'], 'b-', linewidth=0.5, alpha=0.7, label='NIS')
    axes[2].axhline(y=6.63, color='r', linestyle='--', linewidth=1, label='Threshold (6.63)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('NIS', color='b')
    axes[2].set_ylim([0, min(15, np.max(results['NIS']) * 1.1)])
    axes[2].legend(loc='upper left')

    ax2_twin.plot(time, results['robust_weight'], 'g-', linewidth=0.5, alpha=0.7)
    ax2_twin.set_ylabel('Robust Weight', color='g')
    ax2_twin.set_ylim([0, 1.1])

    axes[2].set_title('NIS Gate and Robust Weight')
    axes[2].grid(True, alpha=0.3)

    # 自适应自由度
    axes[3].plot(time, results['nu'], 'm-', linewidth=0.8)
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Student-t ν')
    axes[3].set_title('Adaptive Degrees of Freedom')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"SR-UKF details saved to: {save_path}")
    plt.close()


def print_summary(results_dict: dict):
    """打印汇总表格"""
    print("\n" + "="*70)
    print("                        SUMMARY")
    print("="*70)
    print(f"{'Algorithm':<12} {'RMSE':<14} {'MAE':<14} {'Max Error':<14}")
    print("-"*70)

    for name, (results, metrics) in results_dict.items():
        print(f"{name:<12} {metrics['rmse']:.4f}%{'':<8} {metrics['mae']:.4f}%{'':<8} {metrics['max_error']:.4f}%")

    print("-"*70)

    # 找出最佳算法
    best_rmse = min(results_dict.items(), key=lambda x: x[1][1]['rmse'])
    best_max = min(results_dict.items(), key=lambda x: x[1][1]['max_error'])

    print(f"Best RMSE: {best_rmse[0]} ({best_rmse[1][1]['rmse']:.4f}%)")
    print(f"Best Max Error: {best_max[0]} ({best_max[1][1]['max_error']:.4f}%)")
    print("="*70)


def main():
    """主函数"""
    print("="*70)
    print("       SOC Estimation Algorithms Comparison")
    print("       AEKF vs UKF vs Robust SR-UKF")
    print("="*70)

    # 路径设置
    base_path = Path(__file__).parent
    data_path = base_path / "dataset" / "processed" / "25C_DST_80SOC.csv"
    save_dir = base_path / "dataset" / "processed"

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("Please run process_battery_data.py first.")
        return

    # 加载数据 (SOC: 80% → 10%)
    print(f"\nLoading data from: {data_path}")
    data = load_data(data_path, soc_min=10.0, soc_max=100.0)

    # 运行各算法
    results_dict = {}

    aekf_results, aekf_metrics = run_aekf(data)
    results_dict['AEKF'] = (aekf_results, aekf_metrics)

    ukf_results, ukf_metrics = run_ukf(data)
    results_dict['UKF'] = (ukf_results, ukf_metrics)

    sr_ukf_results, sr_ukf_metrics = run_sr_ukf(data)
    results_dict['SR-UKF'] = (sr_ukf_results, sr_ukf_metrics)

    gst_iaekf_results, gst_iaekf_metrics = run_gst_iaekf(data)
    results_dict['GST-IAEKF'] = (gst_iaekf_results, gst_iaekf_metrics)

    # 绘制对比图
    plot_comparison(data, results_dict, save_dir / "comparison_80_to_10.png")

    # 绘制SR-UKF详细结果
    plot_sr_ukf_details(data, sr_ukf_results, sr_ukf_metrics,
                        save_dir / "SR_UKF_details_80_to_10.png")

    # 打印汇总
    print_summary(results_dict)


if __name__ == "__main__":
    main()
