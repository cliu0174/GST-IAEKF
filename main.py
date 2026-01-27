"""
Main script for SOC Estimation
SOC估计算法主运行脚本

运行方式:
    python main.py

输出:
    - 控制台显示算法性能指标
    - results/graphs/{数据集名}/ 目录下生成结果图
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


# ============================================================================
#                           用户配置区域 (Configuration)
# ============================================================================

# 选择要运行的模型 (可选: "AEKF", "UKF", "SR-UKF", "GST-IAEKF")
# 设置为 None 或 [] 运行所有模型，或指定列表运行特定模型
# 示例:
#   RUN_MODELS = None                    # 运行所有模型
#   RUN_MODELS = ["AEKF"]               # 只运行 AEKF
#   RUN_MODELS = ["AEKF", "UKF"]        # 运行 AEKF 和 UKF
#   RUN_MODELS = ["SR-UKF", "GST-IAEKF"] # 运行 SR-UKF 和 GST-IAEKF
RUN_MODELS = None

# 选择数据文件 (相对于 dataset/processed/ 目录)
# 可用数据集:
#   - "25C_DST_80SOC.csv"    : DST工况, 80%初始SOC
#   - "25C_DST_50SOC.csv"    : DST工况, 50%初始SOC
#   - "25C_FUDS_80SOC.csv"   : FUDS工况, 80%初始SOC
#   - "25C_FUDS_50SOC.csv"   : FUDS工况, 50%初始SOC
#   - "25C_US06_80SOC.csv"   : US06工况, 80%初始SOC
#   - "25C_US06_50SOC.csv"   : US06工况, 50%初始SOC
#   - "25C_BBDST_80SOC.csv"  : BBDST工况, 80%初始SOC
#   - "25C_BBDST_50SOC.csv"  : BBDST工况, 50%初始SOC
DATA_FILE = "25C_DST_80SOC.csv"

# SOC过滤范围 (%)
SOC_MIN = 10.0
SOC_MAX = 100.0

# ============================================================================
#                           配置区域结束
# ============================================================================

# 所有可用模型
ALL_MODELS = ["AEKF", "UKF", "SR-UKF", "GST-IAEKF"]


def load_data(data_path: Path, soc_min: float = 10.0, soc_max: float = 100.0):
    """加载并过滤数据"""
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
        enable_qr_adaptive=False,
        lambda_max=3.0,
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


def get_dataset_name(data_file: str) -> str:
    """从数据文件名提取数据集名称"""
    return Path(data_file).stem


def get_models_to_run() -> list:
    """获取要运行的模型列表"""
    if RUN_MODELS is None or len(RUN_MODELS) == 0:
        return ALL_MODELS.copy()

    valid_models = []
    for model in RUN_MODELS:
        if model in ALL_MODELS:
            valid_models.append(model)
        else:
            print(f"Warning: Unknown model '{model}', skipping...")

    return valid_models


def run_model(model_name: str, data: dict):
    """根据模型名称运行对应算法"""
    model_runners = {
        'AEKF': run_aekf,
        'UKF': run_ukf,
        'SR-UKF': run_sr_ukf,
        'GST-IAEKF': run_gst_iaekf
    }

    if model_name in model_runners:
        return model_runners[model_name](data)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def plot_result(data: dict, model_name: str, results: dict, metrics: dict,
                save_path: Path):
    """绘制单个模型的结果图"""
    time = data['time']
    soc_true = data['soc_true']

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # SOC估计图
    axes[0].plot(time, soc_true, 'b-', linewidth=2, label='True SOC')
    axes[0].plot(time, results['SOC_percent'], 'r--', linewidth=1.5,
                 label=f'{model_name} (RMSE={metrics["rmse"]:.3f}%)')
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('SOC (%)', fontsize=12)
    axes[0].set_title(f'{model_name} SOC Estimation', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 误差图
    axes[1].plot(time, metrics['error'], 'g-', linewidth=1.0)
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('SOC Error (%)', fontsize=12)
    axes[1].set_title(f'Estimation Error (Max={metrics["max_error"]:.3f}%)', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Plot saved to: {save_path}")
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

    if len(results_dict) > 1:
        best_rmse = min(results_dict.items(), key=lambda x: x[1][1]['rmse'])
        best_max = min(results_dict.items(), key=lambda x: x[1][1]['max_error'])
        print(f"Best RMSE: {best_rmse[0]} ({best_rmse[1][1]['rmse']:.4f}%)")
        print(f"Best Max Error: {best_max[0]} ({best_max[1][1]['max_error']:.4f}%)")

    print("="*70)


def main():
    """主函数"""
    models_to_run = get_models_to_run()
    dataset_name = get_dataset_name(DATA_FILE)

    print("="*70)
    print("       SOC Estimation Algorithms")
    print("="*70)
    print(f"  Dataset: {DATA_FILE}")
    print(f"  Models:  {', '.join(models_to_run)}")
    print(f"  SOC Range: {SOC_MIN}% ~ {SOC_MAX}%")
    print("="*70)

    # 路径设置
    base_path = Path(__file__).parent
    data_path = base_path / "dataset" / "processed" / DATA_FILE
    save_dir = base_path / "results" / "graphs" / dataset_name

    save_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("Please run process_battery_data.py first.")
        print("\nAvailable datasets:")
        processed_dir = base_path / "dataset" / "processed"
        if processed_dir.exists():
            for f in processed_dir.glob("*.csv"):
                print(f"  - {f.name}")
        return

    # 加载数据
    print(f"\nLoading data from: {data_path}")
    data = load_data(data_path, soc_min=SOC_MIN, soc_max=SOC_MAX)

    # 运行选定的算法
    results_dict = {}

    for model_name in models_to_run:
        results, metrics = run_model(model_name, data)
        results_dict[model_name] = (results, metrics)

        # 保存结果图
        model_filename = model_name.replace("-", "_")
        plot_result(data, model_name, results, metrics,
                    save_dir / f"{model_filename}_{dataset_name}.png")

    # 打印汇总
    print_summary(results_dict)

    print(f"\n结果已保存到: {save_dir}")


if __name__ == "__main__":
    main()
