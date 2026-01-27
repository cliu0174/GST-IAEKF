"""
OCV-SOC Data Extraction Script
从增量OCV测试数据中提取放电OCV-SOC关系曲线

用法:
    python extract_ocv_soc.py              # 提取所有温度的OCV-SOC数据
    python extract_ocv_soc.py -t 0         # 只提取0C的数据
    python extract_ocv_soc.py -t 25        # 只提取25C的数据
    python extract_ocv_soc.py -t 45        # 只提取45C的数据
"""

import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 电池额定容量 (Ah)
RATED_CAPACITY = 2.0


def extract_discharge_ocv(file_path: Path) -> pd.DataFrame:
    """
    从增量OCV测试文件中提取放电OCV-SOC数据

    Args:
        file_path: Excel文件路径

    Returns:
        包含SOC和OCV列的DataFrame
    """
    print(f"  Reading: {file_path.name}")

    # 读取所有Channel sheet并合并
    xlsx = pd.ExcelFile(file_path)
    dfs = []
    for sheet in xlsx.sheet_names:
        if sheet.startswith('Channel'):
            dfs.append(pd.read_excel(xlsx, sheet_name=sheet))

    if not dfs:
        print(f"  Error: No Channel sheets found in {file_path}")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # 提取放电OCV数据
    discharge_ocv = []
    for cycle in df['Cycle_Index'].unique():
        cycle_data = df[df['Cycle_Index'] == cycle]
        for step in [4, 6]:  # 放电后静置步骤
            step_data = cycle_data[cycle_data['Step_Index'] == step]
            if len(step_data) > 100:
                # 取静置步骤最后50个点的平均值
                last_points = step_data.tail(50)
                ocv = last_points['Voltage(V)'].mean()
                discharge_cap = last_points['Discharge_Capacity(Ah)'].mean()
                charge_cap = last_points['Charge_Capacity(Ah)'].mean()
                net_discharge = discharge_cap - charge_cap
                soc = (RATED_CAPACITY - net_discharge) / RATED_CAPACITY * 100
                discharge_ocv.append({'SOC': soc, 'OCV': ocv})

    result = pd.DataFrame(discharge_ocv).sort_values('SOC').reset_index(drop=True)
    print(f"  Extracted: {len(result)} points")

    return result


def save_ocv_data(df: pd.DataFrame, output_dir: Path, sample_name: str):
    """
    保存OCV-SOC数据到CSV文件并生成曲线图

    Args:
        df: OCV-SOC数据
        output_dir: 输出目录
        sample_name: 样本名称 (用于图表标题)
    """
    if df.empty:
        return

    # 准备保存的数据
    df_save = df.copy()
    df_save.columns = ['SOC_percent', 'OCV_V']
    df_save = df_save.round(4)

    # 保存CSV
    csv_path = output_dir / 'Discharge_OCV_SOC_data.csv'
    df_save.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # 绘制曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(df['SOC'], df['OCV'], 'b-o', markersize=8, linewidth=2)
    plt.xlabel('SOC (%)', fontsize=12)
    plt.ylabel('OCV (V)', fontsize=12)
    plt.title(f'Discharge OCV-SOC Curve ({sample_name})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 105)
    plt.ylim(3.2, 4.3)
    plt.tight_layout()

    fig_path = output_dir / 'Discharge_OCV_SOC_curve.png'
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig_path}")


def save_comparison_plot(sample1_df: pd.DataFrame, sample2_df: pd.DataFrame,
                         output_dir: Path, temp_label: str):
    """
    保存两个样本的对比图

    Args:
        sample1_df: Sample1数据
        sample2_df: Sample2数据
        output_dir: 输出目录
        temp_label: 温度标签 (如 "0C", "25C", "45C")
    """
    if sample1_df.empty and sample2_df.empty:
        return

    plt.figure(figsize=(10, 6))

    if not sample1_df.empty:
        plt.plot(sample1_df['SOC'], sample1_df['OCV'], 'b-o',
                markersize=8, linewidth=2, label='SP20-1 (Sample1)')

    if not sample2_df.empty:
        plt.plot(sample2_df['SOC'], sample2_df['OCV'], 'r-s',
                markersize=8, linewidth=2, label='SP20-3 (Sample2)')

    plt.xlabel('SOC (%)', fontsize=12)
    plt.ylabel('OCV (V)', fontsize=12)
    plt.title(f'Discharge OCV-SOC Curves Comparison ({temp_label})', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 105)
    plt.ylim(3.2, 4.3)
    plt.tight_layout()

    fig_path = output_dir / 'Discharge_OCV_SOC_comparison.png'
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig_path}")


def process_temperature(temp: int, base_path: Path):
    """
    处理指定温度的OCV-SOC数据

    Args:
        temp: 温度 (0, 25, 45)
        base_path: 数据集根目录
    """
    print(f"\n{'='*60}")
    print(f"Processing {temp}C OCV-SOC Data")
    print('='*60)

    # 确定OCV-SOC数据路径
    if temp == 25:
        ocv_dir = base_path / "OCV-SOC"
    else:
        ocv_dir = base_path / f"{temp}C" / "OCV-SOC"

    if not ocv_dir.exists():
        print(f"  Error: OCV-SOC directory not found: {ocv_dir}")
        return

    sample_data = {}

    # 处理每个样本
    for sample_name in ['Sample1', 'Sample2']:
        sample_dir = ocv_dir / sample_name
        if not sample_dir.exists():
            print(f"  Warning: {sample_name} directory not found")
            continue

        print(f"\n[{sample_name}]")

        # 查找Excel文件
        excel_files = list(sample_dir.glob("*.xls*"))
        if not excel_files:
            print(f"  No Excel files found")
            continue

        # 提取OCV-SOC数据
        df = extract_discharge_ocv(excel_files[0])

        if not df.empty:
            # 打印数据范围
            print(f"  SOC range: {df['SOC'].min():.1f}% ~ {df['SOC'].max():.1f}%")
            print(f"  OCV range: {df['OCV'].min():.3f}V ~ {df['OCV'].max():.3f}V")

            # 保存数据和图表
            save_ocv_data(df, sample_dir, f"{temp}C, {sample_name}")
            sample_data[sample_name] = df
        else:
            sample_data[sample_name] = pd.DataFrame()

    # 生成对比图
    if sample_data:
        print(f"\n[Comparison]")
        save_comparison_plot(
            sample_data.get('Sample1', pd.DataFrame()),
            sample_data.get('Sample2', pd.DataFrame()),
            ocv_dir,
            f"{temp}C"
        )


def create_temperature_comparison(base_path: Path):
    """
    创建不同温度下OCV-SOC曲线的对比图
    """
    print(f"\n{'='*60}")
    print("Creating Temperature Comparison Plot")
    print('='*60)

    plt.figure(figsize=(12, 8))
    colors = {0: 'blue', 25: 'green', 45: 'red'}
    markers = {0: 'o', 25: 's', 45: '^'}

    for temp in [0, 25, 45]:
        if temp == 25:
            csv_path = base_path / "OCV-SOC" / "Sample1" / "Discharge_OCV_SOC_data.csv"
        else:
            csv_path = base_path / f"{temp}C" / "OCV-SOC" / "Sample1" / "Discharge_OCV_SOC_data.csv"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            label = f"{temp}C"
            plt.plot(df['SOC_percent'], df['OCV_V'], f'-{markers[temp]}',
                    color=colors[temp], markersize=6, linewidth=2, label=label)
            print(f"  Added {temp}C data ({len(df)} points)")
        else:
            print(f"  Warning: {temp}C data not found")

    plt.xlabel('SOC (%)', fontsize=12)
    plt.ylabel('OCV (V)', fontsize=12)
    plt.title('OCV-SOC Curves at Different Temperatures (Sample1)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 105)
    plt.ylim(3.2, 4.3)
    plt.tight_layout()

    output_path = base_path / "OCV-SOC" / "Temperature_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract OCV-SOC data from incremental OCV tests')
    parser.add_argument('-t', '--temperature', type=int, choices=[0, 25, 45],
                       help='Temperature to process (0, 25, or 45). If not specified, process all.')
    args = parser.parse_args()

    base_path = PROJECT_ROOT / "dataset"

    if args.temperature is not None:
        # 处理指定温度
        process_temperature(args.temperature, base_path)
    else:
        # 处理所有温度
        for temp in [0, 25, 45]:
            process_temperature(temp, base_path)

        # 创建温度对比图
        create_temperature_comparison(base_path)

    print("\n" + "="*60)
    print("OCV-SOC Extraction Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
