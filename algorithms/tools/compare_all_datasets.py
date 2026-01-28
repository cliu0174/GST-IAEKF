"""
一键对比所有数据集的SOC估计算法性能
Compare All Datasets SOC Estimation Performance

运行方式:
    python algorithms/tools/compare_all_datasets.py

输出:
    - 控制台显示所有数据集的汇总对比表格
    - results/graphs/summary/ 目录下生成汇总图表和CSV文件
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
# 设置为 None 或 [] 运行所有模型
RUN_MODELS = None

# 选择要对比的数据集 (设置为 None 对比所有数据集)
# 格式: (温度文件夹, 文件名)
# 可以用元组或列表:
#   RUN_DATASETS = None                              # 所有数据集
#   RUN_DATASETS = [("25C", "DST_50SOC.csv")]       # 单个数据集
#   RUN_DATASETS = [("25C", "DST_50SOC.csv"), ("0C", "DST_50SOC.csv")]  # 多个数据集
# 可选温度: 0C, 25C, 45C
# 可选工况: DST, FUDS, US06, BBDST (各有50SOC和80SOC)
RUN_DATASETS = [("0C", "DST_50SOC.csv"), ("0C", "FUDS_50SOC.csv"), ("0C", "US06_50SOC.csv"), ("0C", "BBDST_50SOC.csv")]

# SOC过滤范围 (%)
SOC_MIN = 10.0
SOC_MAX = 100.0

# ============================================================================
#                           配置区域结束
# ============================================================================

ALL_MODELS = ["AEKF", "UKF", "SR-UKF", "GST-IAEKF"]

# 所有数据集列表: (温度文件夹, 文件名)
ALL_DATASETS = [
    # 0C数据
    ("0C", "DST_80SOC.csv"),
    ("0C", "DST_50SOC.csv"),
    ("0C", "FUDS_80SOC.csv"),
    ("0C", "FUDS_50SOC.csv"),
    ("0C", "US06_80SOC.csv"),
    ("0C", "US06_50SOC.csv"),
    ("0C", "BBDST_80SOC.csv"),
    ("0C", "BBDST_50SOC.csv"),
    # 25C数据
    ("25C", "DST_80SOC.csv"),
    ("25C", "DST_50SOC.csv"),
    ("25C", "FUDS_80SOC.csv"),
    ("25C", "FUDS_50SOC.csv"),
    ("25C", "US06_80SOC.csv"),
    ("25C", "US06_50SOC.csv"),
    ("25C", "BBDST_80SOC.csv"),
    ("25C", "BBDST_50SOC.csv"),
    # 45C数据
    ("45C", "DST_80SOC.csv"),
    ("45C", "DST_50SOC.csv"),
    ("45C", "FUDS_80SOC.csv"),
    ("45C", "FUDS_50SOC.csv"),
    ("45C", "US06_80SOC.csv"),
    ("45C", "US06_50SOC.csv"),
    ("45C", "BBDST_80SOC.csv"),
    ("45C", "BBDST_50SOC.csv"),
]


def load_data(data_path: Path, soc_min: float, soc_max: float):
    """加载并过滤数据"""
    df = pd.read_csv(data_path)
    mask = (df['soc_percent'] >= soc_min) & (df['soc_percent'] <= soc_max)
    df_filtered = df[mask].reset_index(drop=True)

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


def run_aekf(data: dict, temperature: str):
    """运行AEKF算法"""
    initial_soc = data['soc_true'][0] / 100 - 0.1
    # initial_soc = 0.3
    aekf = AEKF(
        initial_soc=initial_soc,
        capacity_Ah=2.0,
        sample_time=1.0,
        use_online_param_id=True,
        adaptive_Q=True,
        temperature=temperature
    )
    results = aekf.estimate_batch(
        data['voltage'], data['current'], data['soc_true'], initial_soc
    )
    soc_est = results['SOC_percent']
    metrics = calculate_metrics(soc_est, data['soc_true'])
    return soc_est, metrics


def run_ukf(data: dict, temperature: str):
    """运行UKF算法"""
    initial_soc = data['soc_true'][0] / 100 - 0.1
    # initial_soc = 0.3
    ukf = UKF(
        initial_soc=initial_soc,
        capacity_Ah=2.0,
        sample_time=1.0,
        use_online_param_id=True,
        temperature=temperature
    )
    results = ukf.estimate_batch(
        data['voltage'], data['current'], data['soc_true'], initial_soc
    )
    soc_est = results['SOC_percent']
    metrics = calculate_metrics(soc_est, data['soc_true'])
    return soc_est, metrics


def run_sr_ukf(data: dict, temperature: str):
    """运行SR-UKF算法"""
    initial_soc = data['soc_true'][0] / 100 - 0.1
    # initial_soc = 0.3
    sr_ukf = RobustSRUKF(
        initial_soc=initial_soc,
        capacity_Ah=2.0,
        sample_time=1.0,
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


def run_gst_iaekf(data: dict, temperature: str):
    """运行GST-IAEKF算法"""
    initial_soc = data['soc_true'][0] / 100 - 0.1
    # initial_soc = 0.3
    gst_iaekf = GSTIAEKF(
        initial_soc=initial_soc,
        capacity_Ah=2.0,
        sample_time=1.0,
        use_online_param_id=True,
        enable_nis_gate=True,
        enable_strong_tracking=True,
        enable_qr_adaptive=False,
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


def run_model(model_name: str, data: dict, temperature: str):
    """运行指定模型，返回 (soc_estimated, metrics)"""
    runners = {
        'AEKF': run_aekf,
        'UKF': run_ukf,
        'SR-UKF': run_sr_ukf,
        'GST-IAEKF': run_gst_iaekf
    }
    return runners[model_name](data, temperature)


def plot_soc_comparison(data: dict, model_results: dict, dataset_name: str, save_path: Path):
    """绘制单个数据集的SOC对比图和误差图"""
    time = data['time']
    soc_true = data['soc_true']

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    colors = {'AEKF': '#1f77b4', 'UKF': '#ff7f0e', 'SR-UKF': '#2ca02c', 'GST-IAEKF': '#d62728'}
    linestyles = {'AEKF': '--', 'UKF': '-.', 'SR-UKF': ':', 'GST-IAEKF': '-'}

    # 上图: SOC对比
    axes[0].plot(time, soc_true, 'b-', linewidth=2, label='True SOC', alpha=0.8)
    for model_name, (soc_est, metrics) in model_results.items():
        axes[0].plot(time, soc_est,
                     color=colors.get(model_name, 'gray'),
                     linestyle=linestyles.get(model_name, '-'),
                     linewidth=1.5,
                     label=f'{model_name} (RMSE={metrics["rmse"]:.3f}%)')

    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('SOC (%)', fontsize=12)
    axes[0].set_title(f'SOC Estimation Comparison - {dataset_name}', fontsize=14)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 下图: 误差对比
    for model_name, (soc_est, metrics) in model_results.items():
        axes[1].plot(time, metrics['error'],
                     color=colors.get(model_name, 'gray'),
                     linewidth=1.0, alpha=0.8,
                     label=f'{model_name} (Max={metrics["max_error"]:.3f}%)')

    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('SOC Error (%)', fontsize=12)
    axes[1].set_title('Estimation Error Comparison', fontsize=14)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"    Saved: {save_path}")
    plt.close()


def plot_comparison_bar(all_results: dict, models: list, save_path: Path):
    """绘制柱状对比图"""
    datasets = list(all_results.keys())
    n_datasets = len(datasets)
    n_models = len(models)

    # 准备数据
    rmse_data = np.zeros((n_datasets, n_models))
    mae_data = np.zeros((n_datasets, n_models))
    max_error_data = np.zeros((n_datasets, n_models))

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            if model in all_results[dataset]:
                rmse_data[i, j] = all_results[dataset][model]['rmse']
                mae_data[i, j] = all_results[dataset][model]['mae']
                max_error_data[i, j] = all_results[dataset][model]['max_error']

    # 数据集标签 (格式已经是 "25C_DST_50SOC")
    dataset_labels = datasets

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x = np.arange(n_datasets)
    width = 0.18
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # RMSE
    for j, model in enumerate(models):
        axes[0].bar(x + j * width, rmse_data[:, j], width, label=model, color=colors[j])
    axes[0].set_xlabel('Dataset')
    axes[0].set_ylabel('RMSE (%)')
    axes[0].set_title('RMSE Comparison')
    axes[0].set_xticks(x + width * (n_models - 1) / 2)
    axes[0].set_xticklabels(dataset_labels, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # MAE
    for j, model in enumerate(models):
        axes[1].bar(x + j * width, mae_data[:, j], width, label=model, color=colors[j])
    axes[1].set_xlabel('Dataset')
    axes[1].set_ylabel('MAE (%)')
    axes[1].set_title('MAE Comparison')
    axes[1].set_xticks(x + width * (n_models - 1) / 2)
    axes[1].set_xticklabels(dataset_labels, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    # Max Error
    for j, model in enumerate(models):
        axes[2].bar(x + j * width, max_error_data[:, j], width, label=model, color=colors[j])
    axes[2].set_xlabel('Dataset')
    axes[2].set_ylabel('Max Error (%)')
    axes[2].set_title('Max Error Comparison')
    axes[2].set_xticks(x + width * (n_models - 1) / 2)
    axes[2].set_xticklabels(dataset_labels, rotation=45, ha='right')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Bar chart saved to: {save_path}")
    plt.close()


def plot_heatmap(all_results: dict, models: list, save_path: Path):
    """绘制热力图"""
    datasets = list(all_results.keys())
    # 数据集标签 (格式已经是 "25C_DST_50SOC")
    dataset_labels = datasets

    # RMSE热力图
    rmse_matrix = np.zeros((len(datasets), len(models)))
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            if model in all_results[dataset]:
                rmse_matrix[i, j] = all_results[dataset][model]['rmse']

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(rmse_matrix, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(models)
    ax.set_yticklabels(dataset_labels)

    # 添加数值标签
    for i in range(len(datasets)):
        for j in range(len(models)):
            text = ax.text(j, i, f'{rmse_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title('RMSE Heatmap (%) - Lower is Better', fontsize=14)
    plt.colorbar(im, ax=ax, label='RMSE (%)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Heatmap saved to: {save_path}")
    plt.close()


def print_summary_table(all_results: dict, models: list):
    """打印汇总表格"""
    print("\n" + "="*90)
    print("                      ALL DATASETS COMPARISON SUMMARY")
    print("="*90)

    # RMSE表格
    print("\n[RMSE (%)]")
    header = f"{'Dataset':<25}"
    for model in models:
        header += f" {model:>12}"
    print(header)
    print("-"*90)

    for dataset, metrics in all_results.items():
        # dataset 格式已经是 "25C_DST_50SOC"
        row = f"  {dataset:<23}"
        for model in models:
            if model in metrics:
                row += f" {metrics[model]['rmse']:>12.4f}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # MAE表格
    print("\n[MAE (%)]")
    print(header)
    print("-"*90)

    for dataset, metrics in all_results.items():
        row = f"  {dataset:<23}"
        for model in models:
            if model in metrics:
                row += f" {metrics[model]['mae']:>12.4f}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # Max Error表格
    print("\n[Max Error (%)]")
    print(header)
    print("-"*90)

    for dataset, metrics in all_results.items():
        row = f"  {dataset:<23}"
        for model in models:
            if model in metrics:
                row += f" {metrics[model]['max_error']:>12.4f}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # 平均值
    print("\n" + "-"*90)
    print("[Average RMSE (%)]")
    row = f"  {'AVERAGE':<18}"
    for model in models:
        rmse_vals = [m[model]['rmse'] for m in all_results.values() if model in m]
        if rmse_vals:
            row += f" {np.mean(rmse_vals):>12.4f}"
        else:
            row += f" {'N/A':>12}"
    print(row)

    print("="*90)

    # 最佳模型
    print("\n[Best Model by Average RMSE]")
    avg_rmse = {}
    for model in models:
        rmse_vals = [m[model]['rmse'] for m in all_results.values() if model in m]
        if rmse_vals:
            avg_rmse[model] = np.mean(rmse_vals)

    if avg_rmse:
        best = min(avg_rmse, key=avg_rmse.get)
        print(f"  {best}: {avg_rmse[best]:.4f}%")

    print("="*90)


def save_to_csv(all_results: dict, models: list, save_path: Path):
    """保存结果到CSV"""
    rows = []
    for dataset, metrics in all_results.items():
        # dataset 格式已经是 "25C_DST_50SOC"
        for model in models:
            if model in metrics:
                rows.append({
                    'Dataset': dataset,
                    'Model': model,
                    'RMSE': metrics[model]['rmse'],
                    'MAE': metrics[model]['mae'],
                    'Max_Error': metrics[model]['max_error']
                })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"  CSV saved to: {save_path}")


def main():
    """主函数"""
    # 确定要运行的模型和数据集
    models = RUN_MODELS if RUN_MODELS else ALL_MODELS
    # 处理数据集配置 (支持 None 或 列表)
    if RUN_DATASETS is None:
        datasets = ALL_DATASETS
    else:
        datasets = RUN_DATASETS

    print("="*70)
    print("       SOC Estimation - All Datasets Comparison")
    print("="*70)
    print(f"  Models: {', '.join(models)}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  SOC Range: {SOC_MIN}% ~ {SOC_MAX}%")
    print("="*70)

    # 输出目录
    save_dir = PROJECT_ROOT / "results" / "graphs" / "summary"
    save_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = PROJECT_ROOT / "dataset" / "processed"

    all_results = {}

    for i, (temp_folder, data_file) in enumerate(datasets):
        data_path = processed_dir / temp_folder / data_file
        # 数据集标识符: 如 "25C_DST_50SOC"
        dataset_key = f"{temp_folder}_{data_file.replace('.csv', '')}"
        print(f"\n[{i+1}/{len(datasets)}] Processing: {dataset_key}")

        if not data_path.exists():
            print(f"  Warning: File not found, skipping...")
            continue

        data = load_data(data_path, SOC_MIN, SOC_MAX)
        print(f"  Data points: {len(data['time'])}")

        all_results[dataset_key] = {}
        model_results = {}  # 存储完整结果用于绘图

        for model in models:
            print(f"    Running {model}...", end=" ")
            soc_est, metrics = run_model(model, data, temp_folder)
            all_results[dataset_key][model] = metrics  # 只存 metrics 用于汇总
            model_results[model] = (soc_est, metrics)  # 存完整结果用于绘图
            print(f"RMSE={metrics['rmse']:.4f}%")

        # 为每个数据集绘制 SOC 对比图和误差图
        dataset_name = dataset_key.replace(".csv", "")
        plot_soc_comparison(data, model_results, dataset_name,
                           save_dir / f"soc_comparison_{dataset_name}.png")

    if not all_results:
        print("Error: No datasets were processed!")
        return

    # 打印汇总
    print_summary_table(all_results, models)

    # 保存汇总图表
    print("\nSaving summary results...")
    plot_comparison_bar(all_results, models, save_dir / "comparison_bar.png")
    plot_heatmap(all_results, models, save_dir / "comparison_heatmap.png")
    save_to_csv(all_results, models, save_dir / "comparison_results.csv")

    print(f"\n所有结果已保存到: {save_dir}")


if __name__ == "__main__":
    main()
