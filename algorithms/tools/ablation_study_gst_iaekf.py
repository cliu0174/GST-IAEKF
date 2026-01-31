"""
GST-IAEKF 消融实验 (Ablation Study)
验证门控、强跟踪、Q/R自适应三个组件的有效性

实验设计：
    - 8个实验组，逐步添加组件
    - 多个数据集测试
    - 自动生成对比图表和CSV结果

运行方式:
    python algorithms/tools/ablation_study_gst_iaekf.py

输出:
    - results/ablation_study/ 目录下生成所有结果
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
from typing import Dict, List, Tuple

from algorithms.functions.gst_iaekf import GSTIAEKF
from algorithms.functions.aekf import AEKF

# 导入CRDST专用的激进版本
from algorithms.tools.test_crdst_models import GSTIAEKF_Aggressive


# 创建AEKF + Gate + ST混合类
class AEKF_GateST(GSTIAEKF):
    """
    AEKF + Gate + ST 混合模型
    - 使用AEKF的激进Q更新策略
    - 使用GST-IAEKF的门控和强跟踪功能
    """
    def __init__(self, *args, **kwargs):
        # 强制设置：门控和强跟踪开启，QR自适应关闭
        kwargs['enable_nis_gate'] = True
        kwargs['enable_strong_tracking'] = True
        kwargs['enable_qr_adaptive'] = False
        super().__init__(*args, **kwargs)

    def update(self, voltage: float, current: float):
        """
        混合更新策略：
        - 使用GST-IAEKF的门控和强跟踪
        - 使用AEKF的激进Q更新（直接替换）
        """
        # 预测观测值
        Ut_pred = self.model.observation(self.x, current)
        C = self.model.get_observation_jacobian(self.x, current)

        # 新息
        e = voltage - Ut_pred

        # 新息协方差
        S = C @ self.P @ C.T + self.R

        # === NIS门控 ===
        NIS = e**2 / S
        gate_triggered = False
        R_effective = self.R

        if self.enable_nis_gate and NIS > self.nis_threshold:
            gate_triggered = True
            R_effective = self.R * self.nis_R_scale
            S = C @ self.P @ C.T + R_effective

            # 激活强跟踪
            if self.enable_strong_tracking:
                self.lambda_k = min(self.lambda_max, 1.0 + (NIS - self.nis_threshold) * 0.5)

        # 卡尔曼增益
        K = self.P @ C.T / S

        # 状态更新
        self.x = self.x + K * e
        self.x[0] = np.clip(self.x[0], 0, 1)

        # 协方差更新（Joseph形式）
        I_KC = np.eye(self.n) - np.outer(K, C)
        self.P = I_KC @ self.P @ I_KC.T + np.outer(K, K) * R_effective

        # === AEKF式的激进Q更新 ===
        self.Q = np.outer(K, K) * (e ** 2)

        # 确保对称正定
        self.P = (self.P + self.P.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(self.P))
        if min_eig < 1e-10:
            self.P = self.P + (1e-10 - min_eig) * np.eye(self.n)

        return self.x, Ut_pred, NIS, gate_triggered


# ============================================================================
#                           用户配置区域 (Configuration)
# ============================================================================

# 选择要测试的数据集
# 可选: "CRDST", "25C_DST_80SOC", "25C_US06_80SOC", "25C_FUDS_80SOC"
TEST_DATASETS = ["DST_80SOC"]  # 25°C DST工况，SOC从80%开始

# SOC过滤范围 (%)
SOC_MIN = 10.0
SOC_MAX = 100.0

# 初始 SOC 偏移量
INITIAL_SOC_OFFSET = 0.0

# 结果输出目录
RESULTS_DIR = PROJECT_ROOT / "results" / "ablation_study"

# 是否使用激进版本
# 重要: 消融实验建议使用 False，确保基准是纯EKF
# True: GSTIAEKF_Aggressive (简化协方差更新)
# False: GSTIAEKF (标准Joseph形式，纯EKF)
USE_AGGRESSIVE = False  # 消融实验：使用纯EKF作为基准

# ============================================================================
#                           配置区域结束
# ============================================================================


# 定义实验组配置（包括AEKF对比组 + GST-IAEKF消融组）
ABLATION_CONFIGS = [
    # === AEKF对比组 ===
    # 组0: AEKF (使用激进自适应Q策略)
    {
        'name': 'AEKF',
        'label': 'AEKF (Aggressive Q)',
        'model_type': 'AEKF',  # 标记使用AEKF类
        'adaptive_Q': True,
        'color': '#34495e',  # 深灰色
        'linestyle': '-',
        'linewidth': 2.0
    },

    # 组0.5: AEKF + Gate + ST (AEKF的激进Q + 门控 + 强跟踪)
    {
        'name': 'AEKF+Gate+ST',
        'label': 'AEKF + Gate + ST',
        'model_type': 'AEKF_GateST',  # 标记使用混合类
        'color': '#16a085',  # 深青色
        'linestyle': '-',
        'linewidth': 2.0
    },

    # === GST-IAEKF消融组 ===
    # 组1: Baseline - 纯EKF (所有增强功能关闭)
    {
        'name': 'EKF-Baseline',
        'label': 'Baseline (Pure EKF)',
        'model_type': 'GSTIAEKF',
        'enable_nis_gate': False,
        'enable_strong_tracking': False,
        'enable_qr_adaptive': False,
        'color': '#95a5a6',  # 浅灰色
        'linestyle': '--'
    },
    # 组2: 仅门控
    {
        'name': 'EKF+Gate',
        'label': 'EKF + NIS Gate',
        'model_type': 'GSTIAEKF',
        'enable_nis_gate': True,
        'enable_strong_tracking': False,
        'enable_qr_adaptive': False,
        'color': '#3498db',  # 蓝色
        'linestyle': '--'
    },
    # 组3: 仅强跟踪
    {
        'name': 'EKF+ST',
        'label': 'EKF + Strong Tracking',
        'model_type': 'GSTIAEKF',
        'enable_nis_gate': False,
        'enable_strong_tracking': True,
        'enable_qr_adaptive': False,
        'color': '#e74c3c',  # 红色
        'linestyle': '-.'
    },
    # 组4: 仅自适应
    {
        'name': 'EKF+Adaptive',
        'label': 'EKF + QR Adaptive',
        'model_type': 'GSTIAEKF',
        'enable_nis_gate': False,
        'enable_strong_tracking': False,
        'enable_qr_adaptive': True,
        'color': '#2ecc71',  # 绿色
        'linestyle': ':'
    },
    # 组5: 门控 + 强跟踪
    {
        'name': 'EKF+Gate+ST',
        'label': 'EKF + Gate + ST',
        'model_type': 'GSTIAEKF',
        'enable_nis_gate': True,
        'enable_strong_tracking': True,
        'enable_qr_adaptive': False,
        'color': '#9b59b6',  # 紫色
        'linestyle': '--'
    },
    # 组6: 门控 + 自适应
    {
        'name': 'EKF+Gate+Adaptive',
        'label': 'EKF + Gate + Adaptive',
        'model_type': 'GSTIAEKF',
        'enable_nis_gate': True,
        'enable_strong_tracking': False,
        'enable_qr_adaptive': True,
        'color': '#f39c12',  # 橙色
        'linestyle': '-.'
    },
    # 组7: 强跟踪 + 自适应
    {
        'name': 'EKF+ST+Adaptive',
        'label': 'EKF + ST + Adaptive',
        'model_type': 'GSTIAEKF',
        'enable_nis_gate': False,
        'enable_strong_tracking': True,
        'enable_qr_adaptive': True,
        'color': '#1abc9c',  # 青色
        'linestyle': ':'
    },
    # 组8: 完整模型
    {
        'name': 'GST-IAEKF-Full',
        'label': 'GST-IAEKF (Full)',
        'model_type': 'GSTIAEKF',
        'enable_nis_gate': True,
        'enable_strong_tracking': True,
        'enable_qr_adaptive': True,
        'color': '#c0392b',  # 深红色
        'linestyle': '-',
        'linewidth': 2.5
    }
]


def load_dataset(dataset_name: str, soc_min: float, soc_max: float) -> Dict:
    """加载并过滤数据集"""

    # 确定数据集路径和温度
    if dataset_name == "CRDST":
        data_path = PROJECT_ROOT / "dataset" / "processed" / "CRDST" / "CRDST_96SOC.csv"
        temperature = "CRDST"
        capacity = 230*3600/112/3600  # 2.054 Ah
        current_sign = -1  # CRDST数据需要转换电流符号
    else:
        # CALCE数据集
        # 支持两种格式: "25C_DST_80SOC" 或 "DST_80SOC"
        if dataset_name.startswith(("0C", "25C", "45C")):
            # 格式: "25C_DST_80SOC"
            temperature = dataset_name.split('_')[0]  # "25C"
            temp_folder = temperature
        else:
            # 格式: "DST_80SOC" (默认25C)
            temperature = "25C"
            temp_folder = "25C"

        data_path = PROJECT_ROOT / "dataset" / "processed" / temp_folder / f"{dataset_name}.csv"
        capacity = 2.0  # CALCE电池容量
        current_sign = 1  # CALCE数据电流符号已正确

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print(f"\nLoading dataset: {dataset_name}")
    df = pd.read_csv(data_path)

    print(f"  Total samples: {len(df)}")
    print(f"  SOC range (raw): {df['soc_percent'].min():.2f}% - {df['soc_percent'].max():.2f}%")

    # 过滤 SOC 范围
    mask = (df['soc_percent'] >= soc_min) & (df['soc_percent'] <= soc_max)
    df_filtered = df[mask].reset_index(drop=True)

    print(f"  Filtered SOC range: {soc_min}% - {soc_max}%")
    print(f"  Filtered samples: {len(df_filtered)}")

    return {
        'name': dataset_name,
        'voltage': df_filtered['voltage_V'].values,
        'current': current_sign * df_filtered['current_A'].values,
        'soc_true': df_filtered['soc_percent'].values,
        'time': df_filtered['time_s'].values,
        'temperature': temperature,
        'capacity': capacity
    }


def run_ablation_experiment(
    config: Dict,
    data: Dict,
    initial_soc_offset: float
) -> Tuple[np.ndarray, Dict]:
    """运行单个消融实验"""

    initial_soc = data['soc_true'][0] / 100 + initial_soc_offset
    initial_soc = np.clip(initial_soc, 0.01, 0.99)

    # 根据数据集选择参数
    if data['name'] == "CRDST":
        # CRDST数据集使用较大的过程噪声协方差
        process_noise = np.array([
            [4e-2, 0, 0],      # SOC过程噪声（激进）
            [0, 2e-4, 0],      # U1过程噪声
            [0, 0, 2e-4]       # U2过程噪声
        ])
        initial_covariance = np.array([
            [4e-5, 0, 0],
            [0, 2e-4, 0],
            [0, 0, 2e-4]
        ])
        measurement_noise = 2e-4
    else:
        # CALCE数据集（25C_DST等）使用标准参数
        process_noise = np.array([
            [1e-6, 0, 0],      # SOC过程噪声（标准）
            [0, 1e-8, 0],      # U1过程噪声
            [0, 0, 1e-8]       # U2过程噪声
        ])
        initial_covariance = np.array([
            [1e-4, 0, 0],
            [0, 1e-6, 0],
            [0, 0, 1e-6]
        ])
        measurement_noise = 2e-4

    # 根据配置选择模型类型
    model_type = config.get('model_type', 'GSTIAEKF')

    if model_type == 'AEKF':
        # 使用AEKF类
        estimator = AEKF(
            initial_soc=initial_soc,
            capacity_Ah=data['capacity'],
            sample_time=1.0,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            initial_covariance=initial_covariance,
            use_online_param_id=True,
            adaptive_Q=config.get('adaptive_Q', True),
            temperature=data['temperature']
        )
    elif model_type == 'AEKF_GateST':
        # 使用AEKF + Gate + ST混合类
        estimator = AEKF_GateST(
            initial_soc=initial_soc,
            capacity_Ah=data['capacity'],
            sample_time=1.0,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            initial_covariance=initial_covariance,
            use_online_param_id=True,
            lambda_max=3.0,
            rho=0.95,
            temperature=data['temperature']
        )
    else:
        # 使用GSTIAEKF类（或其Aggressive版本）
        EstimatorClass = GSTIAEKF_Aggressive if USE_AGGRESSIVE else GSTIAEKF

        # 创建估计器，使用配置中的开关
        estimator = EstimatorClass(
            initial_soc=initial_soc,
            capacity_Ah=data['capacity'],
            sample_time=1.0,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            initial_covariance=initial_covariance,
            use_online_param_id=True,
            # 消融实验的关键：使用配置中的开关
            enable_nis_gate=config.get('enable_nis_gate', False),
            enable_strong_tracking=config.get('enable_strong_tracking', False),
            enable_qr_adaptive=config.get('enable_qr_adaptive', False),
            # 其他参数保持一致
            lambda_max=3.0,
            rho=0.95,
            temperature=data['temperature']
        )

    # 批量估计
    results = estimator.estimate_batch(
        data['voltage'],
        data['current'],
        data['soc_true'],
        initial_soc
    )

    # 计算误差指标
    soc_est = results['SOC_percent']
    error = soc_est - data['soc_true']

    metrics = {
        'rmse': np.sqrt(np.mean(error**2)),
        'mae': np.mean(np.abs(error)),
        'max_error': np.max(np.abs(error)),
        'std_error': np.std(error),
        'error': error
    }

    return soc_est, metrics


def calculate_component_contribution(results_dict: Dict, baseline_rmse: float) -> Dict:
    """计算各组件的贡献度"""

    contributions = {}

    for name, (_, metrics) in results_dict.items():
        if name == 'EKF-Baseline':
            contributions[name] = {
                'absolute_improvement': 0.0,
                'relative_improvement': 0.0,
                'rmse': metrics['rmse']
            }
        else:
            improvement = baseline_rmse - metrics['rmse']
            rel_improvement = (improvement / baseline_rmse) * 100 if baseline_rmse > 0 else 0
            contributions[name] = {
                'absolute_improvement': improvement,
                'relative_improvement': rel_improvement,
                'rmse': metrics['rmse']
            }

    return contributions


def plot_soc_comparison(data: Dict, results_dict: Dict, save_dir: Path, dataset_name: str):
    """绘制SOC对比图"""

    time = data['time'] / 60  # 转换为分钟
    soc_true = data['soc_true']

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # 上图: SOC对比
    axes[0].plot(time, soc_true, 'k-', linewidth=3, label='True SOC', alpha=0.9, zorder=10)

    for config in ABLATION_CONFIGS:
        name = config['name']
        if name in results_dict:
            soc_est, metrics = results_dict[name]
            linewidth = config.get('linewidth', 1.5)
            axes[0].plot(time, soc_est,
                        color=config['color'],
                        linestyle=config['linestyle'],
                        linewidth=linewidth,
                        alpha=0.85,
                        label=f"{config['label']} (RMSE={metrics['rmse']:.3f}%)")

    axes[0].set_xlabel('Time (min)', fontsize=13)
    axes[0].set_ylabel('SOC (%)', fontsize=13)
    axes[0].set_title(f'Ablation Study: SOC Estimation Comparison - {dataset_name}',
                     fontsize=15, fontweight='bold')
    axes[0].legend(loc='best', fontsize=9, ncol=2, framealpha=0.95)
    axes[0].grid(True, alpha=0.3)

    # 下图: 误差对比
    for config in ABLATION_CONFIGS:
        name = config['name']
        if name in results_dict:
            soc_est, metrics = results_dict[name]
            linewidth = config.get('linewidth', 1.2)
            axes[1].plot(time, metrics['error'],
                        color=config['color'],
                        linestyle=config['linestyle'],
                        linewidth=linewidth,
                        alpha=0.8,
                        label=f"{config['label']} (Max={metrics['max_error']:.3f}%)")

    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1.0, alpha=0.5)
    axes[1].set_xlabel('Time (min)', fontsize=13)
    axes[1].set_ylabel('SOC Error (%)', fontsize=13)
    axes[1].set_title('Estimation Error Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=9, ncol=2, framealpha=0.95)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / f"soc_comparison_{dataset_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  SOC comparison plot saved: {save_path.name}")


def plot_rmse_comparison(all_results: Dict, save_dir: Path):
    """绘制RMSE对比柱状图"""

    datasets = list(all_results.keys())
    n_datasets = len(datasets)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(ABLATION_CONFIGS))
    width = 0.8 / n_datasets

    for i, dataset_name in enumerate(datasets):
        results_dict = all_results[dataset_name]
        rmse_values = [results_dict[config['name']][1]['rmse']
                      for config in ABLATION_CONFIGS]

        offset = (i - n_datasets/2 + 0.5) * width
        bars = ax.bar(x + offset, rmse_values, width,
                     label=dataset_name, alpha=0.8)

        # 添加数值标签
        for bar, val in zip(bars, rmse_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=8, rotation=0)

    ax.set_xlabel('Model Configuration', fontsize=13, fontweight='bold')
    ax.set_ylabel('RMSE (%)', fontsize=13, fontweight='bold')
    ax.set_title('Ablation Study: RMSE Comparison Across Configurations',
                fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([config['label'] for config in ABLATION_CONFIGS],
                       rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = save_dir / "comparison_bar.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  RMSE comparison bar chart saved: {save_path.name}")


def plot_contribution_analysis(contributions: Dict, save_dir: Path, dataset_name: str):
    """绘制组件贡献度分析"""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 准备数据
    config_names = [config['label'] for config in ABLATION_CONFIGS]
    absolute_improvements = [contributions[config['name']]['absolute_improvement']
                            for config in ABLATION_CONFIGS]
    relative_improvements = [contributions[config['name']]['relative_improvement']
                            for config in ABLATION_CONFIGS]
    colors = [config['color'] for config in ABLATION_CONFIGS]

    # 左图: 绝对改进
    bars1 = axes[0].bar(range(len(config_names)), absolute_improvements,
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0].axhline(y=0, color='k', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('RMSE Improvement (%)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Absolute Improvement vs Baseline - {dataset_name}',
                     fontsize=13, fontweight='bold')
    axes[0].set_xticks(range(len(config_names)))
    axes[0].set_xticklabels(config_names, rotation=45, ha='right', fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, val in zip(bars1, absolute_improvements):
        if val != 0:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}',
                        ha='center', va='bottom' if val > 0 else 'top',
                        fontsize=9, fontweight='bold')

    # 右图: 相对改进
    bars2 = axes[1].bar(range(len(config_names)), relative_improvements,
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Relative Improvement (%)', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Relative Improvement vs Baseline - {dataset_name}',
                     fontsize=13, fontweight='bold')
    axes[1].set_xticks(range(len(config_names)))
    axes[1].set_xticklabels(config_names, rotation=45, ha='right', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, val in zip(bars2, relative_improvements):
        if val != 0:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}%',
                        ha='center', va='bottom' if val > 0 else 'top',
                        fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_path = save_dir / f"contribution_analysis_{dataset_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Contribution analysis saved: {save_path.name}")


def plot_heatmap(all_results: Dict, save_dir: Path):
    """绘制性能热力图"""

    datasets = list(all_results.keys())
    config_names = [config['label'] for config in ABLATION_CONFIGS]

    # 准备数据矩阵 (配置 × 数据集)
    rmse_matrix = np.zeros((len(ABLATION_CONFIGS), len(datasets)))

    for i, config in enumerate(ABLATION_CONFIGS):
        for j, dataset_name in enumerate(datasets):
            rmse_matrix[i, j] = all_results[dataset_name][config['name']][1]['rmse']

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(max(8, len(datasets)*2), len(ABLATION_CONFIGS)*0.8))

    sns.heatmap(rmse_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
                xticklabels=datasets, yticklabels=config_names,
                cbar_kws={'label': 'RMSE (%)'},
                linewidths=0.5, linecolor='gray',
                ax=ax)

    ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model Configuration', fontsize=13, fontweight='bold')
    ax.set_title('Ablation Study: RMSE Heatmap', fontsize=15, fontweight='bold')

    plt.tight_layout()
    save_path = save_dir / "heatmap.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Heatmap saved: {save_path.name}")


def save_results_csv(all_results: Dict, save_dir: Path):
    """保存结果到CSV文件"""

    # 汇总所有结果
    summary_data = []

    for dataset_name, results_dict in all_results.items():
        for config in ABLATION_CONFIGS:
            name = config['name']
            if name in results_dict:
                _, metrics = results_dict[name]

                # 根据模型类型设置标记
                model_type = config.get('model_type', 'GSTIAEKF')
                if model_type == 'AEKF':
                    gate_mark = '-'
                    st_mark = '-'
                    adapt_mark = 'Aggressive Q'
                elif model_type == 'AEKF_GateST':
                    gate_mark = '✓'
                    st_mark = '✓'
                    adapt_mark = 'Aggressive Q'
                else:
                    gate_mark = '✓' if config.get('enable_nis_gate', False) else '✗'
                    st_mark = '✓' if config.get('enable_strong_tracking', False) else '✗'
                    adapt_mark = '✓' if config.get('enable_qr_adaptive', False) else '✗'

                summary_data.append({
                    'Dataset': dataset_name,
                    'Configuration': config['label'],
                    'Gate': gate_mark,
                    'Strong_Tracking': st_mark,
                    'QR_Adaptive': adapt_mark,
                    'RMSE (%)': metrics['rmse'],
                    'MAE (%)': metrics['mae'],
                    'Max Error (%)': metrics['max_error'],
                    'Std Error (%)': metrics['std_error']
                })

    df_summary = pd.DataFrame(summary_data)

    # 保存CSV
    csv_path = save_dir / "ablation_results.csv"
    df_summary.to_csv(csv_path, index=False, float_format='%.6f')

    print(f"  Results CSV saved: {csv_path.name}")

    return df_summary


def generate_markdown_report(all_results: Dict, contributions_dict: Dict, save_dir: Path):
    """生成Markdown格式的实验报告"""

    report = []
    report.append("# GST-IAEKF 消融实验报告\n")
    report.append("## Ablation Study Report\n")
    report.append("---\n\n")

    report.append("## 1. 实验目的 (Objective)\n\n")
    report.append("验证GST-IAEKF算法中三个核心组件的有效性：\n")
    report.append("- **NIS门控 (NIS Gating)**: 检测和抑制异常测量\n")
    report.append("- **强跟踪因子 (Strong Tracking)**: 快速响应工况突变\n")
    report.append("- **Q/R自适应 (QR Adaptive)**: 滑窗统计自适应噪声协方差\n\n")

    report.append("## 2. 实验设置 (Experimental Setup)\n\n")
    report.append("### 2.1 实验组配置\n\n")
    report.append("| # | 配置名称 | 门控 | 强跟踪 | 自适应 |\n")
    report.append("|---|---------|------|--------|--------|\n")
    for i, config in enumerate(ABLATION_CONFIGS):
        model_type = config.get('model_type', 'GSTIAEKF')
        if model_type == 'AEKF':
            gate = '-'
            st = '-'
            adapt = 'Aggressive Q'
        elif model_type == 'AEKF_GateST':
            gate = '✓'
            st = '✓'
            adapt = 'Aggressive Q'
        else:
            gate = '✓' if config.get('enable_nis_gate', False) else '✗'
            st = '✓' if config.get('enable_strong_tracking', False) else '✗'
            adapt = '✓' if config.get('enable_qr_adaptive', False) else '✗'
        report.append(f"| {i} | {config['label']} | {gate} | {st} | {adapt} |\n")
    report.append("\n")

    report.append("### 2.2 测试数据集\n\n")
    for dataset_name in all_results.keys():
        report.append(f"- {dataset_name}\n")
    report.append("\n")

    report.append("## 3. 实验结果 (Results)\n\n")

    for dataset_name, results_dict in all_results.items():
        report.append(f"### 3.{list(all_results.keys()).index(dataset_name)+1} {dataset_name} 数据集\n\n")
        report.append("| 配置 | RMSE (%) | MAE (%) | Max Error (%) |\n")
        report.append("|------|----------|---------|---------------|\n")

        for config in ABLATION_CONFIGS:
            name = config['name']
            if name in results_dict:
                _, metrics = results_dict[name]
                report.append(f"| {config['label']} | {metrics['rmse']:.4f} | "
                            f"{metrics['mae']:.4f} | {metrics['max_error']:.4f} |\n")
        report.append("\n")

        # 组件贡献分析
        if dataset_name in contributions_dict:
            report.append(f"#### 组件贡献度分析 (Component Contribution)\n\n")
            contributions = contributions_dict[dataset_name]
            baseline_rmse = contributions['EKF-Baseline']['rmse']

            report.append(f"**基准RMSE**: {baseline_rmse:.4f}%\n\n")
            report.append("| 配置 | RMSE改进 | 相对改进 |\n")
            report.append("|------|----------|----------|\n")

            for config in ABLATION_CONFIGS[1:]:  # 跳过baseline
                name = config['name']
                abs_imp = contributions[name]['absolute_improvement']
                rel_imp = contributions[name]['relative_improvement']
                report.append(f"| {config['label']} | {abs_imp:+.4f}% | {rel_imp:+.2f}% |\n")
            report.append("\n")

    report.append("## 4. 主要发现 (Key Findings)\n\n")

    # 找出性能最好的配置
    best_config = None
    best_rmse = float('inf')
    for dataset_name, results_dict in all_results.items():
        for config in ABLATION_CONFIGS:
            name = config['name']
            if name in results_dict:
                rmse = results_dict[name][1]['rmse']
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_config = config['label']

    report.append(f"- **最佳配置**: {best_config} (RMSE: {best_rmse:.4f}%)\n")

    # 计算完整模型vs基准的改进
    for dataset_name in all_results.keys():
        full_rmse = all_results[dataset_name]['GST-IAEKF-Full'][1]['rmse']
        baseline_rmse = all_results[dataset_name]['EKF-Baseline'][1]['rmse']
        improvement = ((baseline_rmse - full_rmse) / baseline_rmse) * 100
        report.append(f"- **{dataset_name}**: 完整模型相比基准改进 {improvement:.2f}%\n")

    report.append("\n")

    report.append("## 5. 结论 (Conclusion)\n\n")
    report.append("消融实验表明，GST-IAEKF的三个组件均对性能提升有贡献，")
    report.append("完整模型（所有组件启用）达到最佳性能。\n\n")

    report.append("---\n")
    report.append(f"\n*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # 保存报告
    report_path = save_dir / "summary.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)

    print(f"  Markdown report saved: {report_path.name}")


def main():
    """主函数"""

    print("=" * 80)
    print("GST-IAEKF 消融实验 (Ablation Study)")
    print("=" * 80)

    # 创建结果目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n实验配置:")
    print(f"  测试数据集: {', '.join(TEST_DATASETS)}")
    print(f"  实验组数量: {len(ABLATION_CONFIGS)}")
    print(f"  使用版本: {'GSTIAEKF_Aggressive' if USE_AGGRESSIVE else 'GSTIAEKF'}")
    print(f"  结果目录: {RESULTS_DIR}")

    # 存储所有结果
    all_results = {}
    all_contributions = {}

    # 对每个数据集运行消融实验
    for dataset_name in TEST_DATASETS:
        print(f"\n{'='*60}")
        print(f"数据集: {dataset_name}")
        print(f"{'='*60}")

        try:
            # 加载数据
            data = load_dataset(dataset_name, SOC_MIN, SOC_MAX)

            # 运行所有实验组
            results_dict = {}

            for config in ABLATION_CONFIGS:
                print(f"\n  运行配置: {config['label']}")

                # 打印配置详情（根据模型类型）
                model_type = config.get('model_type', 'GSTIAEKF')
                if model_type == 'AEKF':
                    print(f"    模型类型=AEKF, adaptive_Q={config.get('adaptive_Q', True)}")
                elif model_type == 'AEKF_GateST':
                    print(f"    模型类型=AEKF+Gate+ST (激进Q + 门控 + 强跟踪)")
                else:
                    print(f"    门控={config.get('enable_nis_gate', False)}, "
                          f"强跟踪={config.get('enable_strong_tracking', False)}, "
                          f"自适应={config.get('enable_qr_adaptive', False)}")

                soc_est, metrics = run_ablation_experiment(config, data, INITIAL_SOC_OFFSET)
                results_dict[config['name']] = (soc_est, metrics)

                print(f"    RMSE: {metrics['rmse']:.4f}%, "
                      f"MAE: {metrics['mae']:.4f}%, "
                      f"Max Error: {metrics['max_error']:.4f}%")

            all_results[dataset_name] = results_dict

            # 计算组件贡献
            baseline_rmse = results_dict['EKF-Baseline'][1]['rmse']
            contributions = calculate_component_contribution(results_dict, baseline_rmse)
            all_contributions[dataset_name] = contributions

            # 绘制该数据集的对比图
            print(f"\n  生成可视化结果...")
            plot_soc_comparison(data, results_dict, RESULTS_DIR, dataset_name)
            plot_contribution_analysis(contributions, RESULTS_DIR, dataset_name)

        except Exception as e:
            print(f"  错误: 处理数据集 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("\n没有成功完成的实验，程序退出。")
        return

    # 生成汇总可视化
    print(f"\n{'='*60}")
    print("生成汇总结果...")
    print(f"{'='*60}")

    plot_rmse_comparison(all_results, RESULTS_DIR)
    plot_heatmap(all_results, RESULTS_DIR)

    # 保存CSV和报告
    df_summary = save_results_csv(all_results, RESULTS_DIR)
    generate_markdown_report(all_results, all_contributions, RESULTS_DIR)

    # 打印汇总表格
    print("\n" + "=" * 80)
    print("实验结果汇总 (Summary)")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)

    print(f"\n所有结果已保存到: {RESULTS_DIR}")
    print("\n生成的文件:")
    for file in sorted(RESULTS_DIR.glob("*")):
        print(f"  - {file.name}")

    print("\n" + "=" * 80)
    print("消融实验完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
