"""
重新处理0C和45C的OCV-SOC数据
按照25C的处理方式，保留10个采样点，去除异常点

处理规则:
- 0C: 去掉100 SOC处的点 (最后一个点101.06% SOC)
- 45C: 去掉0 SOC处的点 (第一个点0.72% SOC)
- 保留10个采样点
- 对Discharge数据进行5阶多项式拟合
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent


def polynomial_5th(soc, a5, a4, a3, a2, a1, a0):
    """5阶多项式: OCV = a5*SOC^5 + a4*SOC^4 + a3*SOC^3 + a2*SOC^2 + a1*SOC + a0"""
    return a5 * soc**5 + a4 * soc**4 + a3 * soc**3 + a2 * soc**2 + a1 * soc + a0


def process_0c_data():
    """处理0C数据：去掉最后一个点（101% SOC）"""
    print("\n" + "="*70)
    print("Processing 0C OCV-SOC Data (Removing 101% SOC point)")
    print("="*70)

    # 读取原始数据
    input_path = PROJECT_ROOT / "dataset" / "0C" / "OCV-SOC" / "Sample1" / "Discharge_OCV_SOC_data.csv"
    df = pd.read_csv(input_path)

    print(f"\nOriginal data: {len(df)} points")
    print(f"SOC range: {df['SOC_percent'].min():.2f}% ~ {df['SOC_percent'].max():.2f}%")
    print(f"OCV range: {df['OCV_V'].min():.4f}V ~ {df['OCV_V'].max():.4f}V")

    # 去掉最后一个点（101% SOC）
    df_filtered = df[df['SOC_percent'] <= 100].copy()

    print(f"\nFiltered data: {len(df_filtered)} points")
    print(f"SOC range: {df_filtered['SOC_percent'].min():.2f}% ~ {df_filtered['SOC_percent'].max():.2f}%")
    print(f"OCV range: {df_filtered['OCV_V'].min():.4f}V ~ {df_filtered['OCV_V'].max():.4f}V")

    return df_filtered


def process_45c_data():
    """处理45C数据：去掉第一个点（0.7% SOC）"""
    print("\n" + "="*70)
    print("Processing 45C OCV-SOC Data (Removing 0.7% SOC point)")
    print("="*70)

    # 读取原始数据
    input_path = PROJECT_ROOT / "dataset" / "45C" / "OCV-SOC" / "Sample1" / "Discharge_OCV_SOC_data.csv"
    df = pd.read_csv(input_path)

    print(f"\nOriginal data: {len(df)} points")
    print(f"SOC range: {df['SOC_percent'].min():.2f}% ~ {df['SOC_percent'].max():.2f}%")
    print(f"OCV range: {df['OCV_V'].min():.4f}V ~ {df['OCV_V'].max():.4f}V")

    # 去掉第一个点（0.7% SOC）
    df_filtered = df[df['SOC_percent'] >= 10].copy()

    print(f"\nFiltered data: {len(df_filtered)} points")
    print(f"SOC range: {df_filtered['SOC_percent'].min():.2f}% ~ {df_filtered['SOC_percent'].max():.2f}%")
    print(f"OCV range: {df_filtered['OCV_V'].min():.4f}V ~ {df_filtered['OCV_V'].max():.4f}V")

    return df_filtered


def fit_polynomial(df, temp_label):
    """对数据进行5阶多项式拟合"""
    print(f"\n[Fitting 5th Order Polynomial for {temp_label}]")

    # 归一化SOC到0-1
    soc_normalized = df['SOC_percent'].values / 100.0
    ocv = df['OCV_V'].values

    # 拟合5阶多项式
    popt, pcov = curve_fit(polynomial_5th, soc_normalized, ocv)
    a5, a4, a3, a2, a1, a0 = popt

    print(f"Fitted coefficients:")
    print(f"  a5 = {a5:.10f}")
    print(f"  a4 = {a4:.10f}")
    print(f"  a3 = {a3:.10f}")
    print(f"  a2 = {a2:.10f}")
    print(f"  a1 = {a1:.10f}")
    print(f"  a0 = {a0:.10f}")

    # 计算拟合误差
    ocv_fitted = polynomial_5th(soc_normalized, *popt)
    residuals = ocv - ocv_fitted
    rmse = np.sqrt(np.mean(residuals**2))
    max_error = np.max(np.abs(residuals))

    print(f"Fitting error:")
    print(f"  RMSE: {rmse*1000:.4f} mV")
    print(f"  Max error: {max_error*1000:.4f} mV")

    return popt, ocv_fitted


def save_data_and_plots(df, popt, ocv_fitted, temp_label, temp_folder):
    """保存处理后的数据和图表"""
    output_dir = PROJECT_ROOT / "dataset" / temp_folder / "OCV-SOC" / "Sample1"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 保存处理后的OCV-SOC数据（新文件名）
    csv_path = output_dir / "Discharge_OCV_SOC_data_filtered.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved filtered data: {csv_path}")

    # 2. 保存多项式系数（新文件名）
    coeff_df = pd.DataFrame({
        'Coefficient': ['a0', 'a1', 'a2', 'a3', 'a4', 'a5'],
        'Value': [popt[5], popt[4], popt[3], popt[2], popt[1], popt[0]]
    })
    coeff_path = output_dir / "5th_order_polynomial_coefficients_filtered.csv"
    coeff_df.to_csv(coeff_path, index=False)
    print(f"  Saved coefficients: {coeff_path}")

    # 3. 绘制拟合对比图
    plt.figure(figsize=(12, 6))

    soc_percent = df['SOC_percent'].values
    ocv = df['OCV_V'].values
    soc_normalized = soc_percent / 100.0

    # 生成平滑曲线用于绘图
    soc_smooth = np.linspace(soc_normalized.min(), soc_normalized.max(), 200)
    ocv_smooth = polynomial_5th(soc_smooth, *popt)

    plt.plot(soc_percent, ocv, 'ro', markersize=10, label='Measured Data', zorder=3)
    plt.plot(soc_smooth * 100, ocv_smooth, 'b-', linewidth=2, label='5th Order Polynomial Fit', zorder=2)

    plt.xlabel('SOC (%)', fontsize=14)
    plt.ylabel('OCV (V)', fontsize=14)
    plt.title(f'5th Order Polynomial Fitting - {temp_label} (Filtered, 10 points)', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(soc_percent.min() - 5, soc_percent.max() + 5)
    plt.tight_layout()

    fig_path = output_dir / "5th_order_polynomial_fit_filtered.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {fig_path}")

    # 4. 绘制残差图
    residuals = ocv - ocv_fitted

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(soc_percent, residuals * 1000, 'ro-', linewidth=2, markersize=8)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('SOC (%)', fontsize=12)
    plt.ylabel('Residual (mV)', fontsize=12)
    plt.title(f'Fitting Residuals - {temp_label}', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(residuals * 1000, bins=15, edgecolor='black', alpha=0.7)
    plt.xlabel('Residual (mV)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Residual Distribution', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    residual_path = output_dir / "fitting_residuals_filtered.png"
    plt.savefig(residual_path, dpi=150)
    plt.close()
    print(f"  Saved residual plot: {residual_path}")


def create_comparison_plot():
    """创建新旧数据对比图"""
    print("\n" + "="*70)
    print("Creating Comparison Plots (Original vs Filtered)")
    print("="*70)

    for temp_folder, temp_label in [("0C", "0°C"), ("45C", "45°C")]:
        print(f"\n[{temp_label}]")

        sample_dir = PROJECT_ROOT / "dataset" / temp_folder / "OCV-SOC" / "Sample1"

        # 读取原始数据和过滤后的数据
        original_path = sample_dir / "Discharge_OCV_SOC_data.csv"
        filtered_path = sample_dir / "Discharge_OCV_SOC_data_filtered.csv"

        if not original_path.exists() or not filtered_path.exists():
            print(f"  Warning: Data files not found")
            continue

        df_original = pd.read_csv(original_path)
        df_filtered = pd.read_csv(filtered_path)

        # 绘制对比图
        plt.figure(figsize=(12, 6))

        plt.plot(df_original['SOC_percent'], df_original['OCV_V'],
                'ro-', markersize=8, linewidth=2, label='Original (11 points)', alpha=0.6)
        plt.plot(df_filtered['SOC_percent'], df_filtered['OCV_V'],
                'bs-', markersize=10, linewidth=2, label='Filtered (10 points)', zorder=3)

        plt.xlabel('SOC (%)', fontsize=14)
        plt.ylabel('OCV (V)', fontsize=14)
        plt.title(f'OCV-SOC Data Comparison - {temp_label}', fontsize=16)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = sample_dir / "comparison_original_vs_filtered.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  Saved: {fig_path}")


def main():
    print("="*70)
    print("Reprocessing 0C and 45C OCV-SOC Data")
    print("="*70)

    # 处理0C数据
    df_0c = process_0c_data()
    popt_0c, ocv_fitted_0c = fit_polynomial(df_0c, "0°C")
    save_data_and_plots(df_0c, popt_0c, ocv_fitted_0c, "0°C", "0C")

    # 处理45C数据
    df_45c = process_45c_data()
    popt_45c, ocv_fitted_45c = fit_polynomial(df_45c, "45°C")
    save_data_and_plots(df_45c, popt_45c, ocv_fitted_45c, "45°C", "45C")

    # 创建对比图
    create_comparison_plot()

    print("\n" + "="*70)
    print("Reprocessing Complete!")
    print("="*70)
    print("\nGenerated files (in dataset/{temp}C/OCV-SOC/Sample1/):")
    print("  - Discharge_OCV_SOC_data_filtered.csv")
    print("  - 5th_order_polynomial_coefficients_filtered.csv")
    print("  - 5th_order_polynomial_fit_filtered.png")
    print("  - fitting_residuals_filtered.png")
    print("  - comparison_original_vs_filtered.png")
    print("\nOriginal files are preserved.")
    print("="*70)


if __name__ == "__main__":
    main()
