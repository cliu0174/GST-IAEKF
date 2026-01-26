"""
Test script for AEKF and UKF SOC estimation algorithms
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import with absolute path handling
from algorithms.functions.battery_model import BatteryModel2RC
from algorithms.functions.ffrls import FFRLS
from algorithms.functions.aekf import AEKF
from algorithms.functions.ukf import UKF
from algorithms.functions.sr_ukf import RobustSRUKF


def test_aekf(data_path: Path, save_dir: Path):
    """Test AEKF algorithm"""
    print("\n" + "="*50)
    print("Testing AEKF Algorithm")
    print("="*50)

    df = pd.read_csv(data_path)
    voltage = df['voltage_V'].values
    current = df['current_A'].values
    soc_true = df['soc_percent'].values
    time = df['time_s'].values

    # Create AEKF estimator
    initial_soc = soc_true[0] / 100
    aekf = AEKF(
        initial_soc=initial_soc,
        capacity_Ah=2.0,
        sample_time=1.0,
        use_online_param_id=True,
        adaptive_Q=True
    )

    # Batch estimation
    print("Running AEKF estimation...")
    results = aekf.estimate_batch(voltage, current, soc_true, initial_soc)

    # Calculate errors
    soc_error = results['SOC_percent'] - soc_true
    rmse = np.sqrt(np.mean(soc_error**2))
    mae = np.mean(np.abs(soc_error))
    max_error = np.max(np.abs(soc_error))

    print(f"\nAEKF Results:")
    print(f"  RMSE: {rmse:.4f}%")
    print(f"  MAE: {mae:.4f}%")
    print(f"  Max Error: {max_error:.4f}%")

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(time, soc_true, 'b-', linewidth=1, label='True SOC')
    axes[0].plot(time, results['SOC_percent'], 'r--', linewidth=1, label='AEKF Estimated')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('SOC (%)')
    axes[0].set_title('AEKF SOC Estimation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, soc_error, 'g-', linewidth=0.8)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('SOC Error (%)')
    axes[1].set_title(f'SOC Estimation Error (RMSE={rmse:.4f}%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    axes[2].plot(time, voltage, 'b-', linewidth=0.8, label='Measured')
    axes[2].plot(time, results['Ut_pred'], 'r--', linewidth=0.8, label='Predicted')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Voltage (V)')
    axes[2].set_title('Terminal Voltage')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "AEKF_results.png"
    plt.savefig(save_path, dpi=150)
    print(f"Results saved to: {save_path}")
    plt.close()

    return results, {'rmse': rmse, 'mae': mae, 'max_error': max_error}


def test_ukf(data_path: Path, save_dir: Path):
    """Test UKF algorithm"""
    print("\n" + "="*50)
    print("Testing UKF Algorithm")
    print("="*50)

    df = pd.read_csv(data_path)
    voltage = df['voltage_V'].values
    current = df['current_A'].values
    soc_true = df['soc_percent'].values
    time = df['time_s'].values

    # Create UKF estimator
    initial_soc = soc_true[0] / 100
    ukf = UKF(
        initial_soc=initial_soc,
        capacity_Ah=2.0,
        sample_time=1.0,
        use_online_param_id=True
    )

    # Batch estimation
    print("Running UKF estimation...")
    results = ukf.estimate_batch(voltage, current, soc_true, initial_soc)

    # Calculate errors
    soc_error = results['SOC_percent'] - soc_true
    rmse = np.sqrt(np.mean(soc_error**2))
    mae = np.mean(np.abs(soc_error))
    max_error = np.max(np.abs(soc_error))

    print(f"\nUKF Results:")
    print(f"  RMSE: {rmse:.4f}%")
    print(f"  MAE: {mae:.4f}%")
    print(f"  Max Error: {max_error:.4f}%")

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(time, soc_true, 'b-', linewidth=1, label='True SOC')
    axes[0].plot(time, results['SOC_percent'], 'r--', linewidth=1, label='UKF Estimated')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('SOC (%)')
    axes[0].set_title('UKF SOC Estimation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, soc_error, 'g-', linewidth=0.8)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('SOC Error (%)')
    axes[1].set_title(f'SOC Estimation Error (RMSE={rmse:.4f}%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    axes[2].plot(time, voltage, 'b-', linewidth=0.8, label='Measured')
    axes[2].plot(time, results['Ut_pred'], 'r--', linewidth=0.8, label='Predicted')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Voltage (V)')
    axes[2].set_title('Terminal Voltage')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "UKF_results.png"
    plt.savefig(save_path, dpi=150)
    print(f"Results saved to: {save_path}")
    plt.close()

    return results, {'rmse': rmse, 'mae': mae, 'max_error': max_error}


def test_sr_ukf(data_path: Path, save_dir: Path):
    """Test Robust SR-UKF algorithm"""
    print("\n" + "="*50)
    print("Testing Robust SR-UKF Algorithm")
    print("="*50)

    df = pd.read_csv(data_path)
    voltage = df['voltage_V'].values
    current = df['current_A'].values
    soc_true = df['soc_percent'].values
    time = df['time_s'].values

    # Create SR-UKF estimator
    initial_soc = soc_true[0] / 100
    sr_ukf = RobustSRUKF(
        initial_soc=initial_soc,
        capacity_Ah=2.0,
        sample_time=1.0,
        use_online_param_id=True,
        enable_nis_gate=True,
        enable_student_t=True,
        adaptive_nu=True
    )

    # Batch estimation
    print("Running Robust SR-UKF estimation...")
    results = sr_ukf.estimate_batch(voltage, current, soc_true, initial_soc)

    # Calculate errors
    soc_error = results['SOC_percent'] - soc_true
    rmse = np.sqrt(np.mean(soc_error**2))
    mae = np.mean(np.abs(soc_error))
    max_error = np.max(np.abs(soc_error))

    # NIS statistics
    nis_triggered = np.sum(results['NIS'] > 6.63)

    print(f"\nRobust SR-UKF Results:")
    print(f"  RMSE: {rmse:.4f}%")
    print(f"  MAE: {mae:.4f}%")
    print(f"  Max Error: {max_error:.4f}%")
    print(f"  NIS Gate Triggered: {nis_triggered} times ({100*nis_triggered/len(soc_true):.2f}%)")

    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

    axes[0].plot(time, soc_true, 'b-', linewidth=1, label='True SOC')
    axes[0].plot(time, results['SOC_percent'], 'r--', linewidth=1, label='SR-UKF Estimated')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('SOC (%)')
    axes[0].set_title('Robust SR-UKF SOC Estimation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, soc_error, 'g-', linewidth=0.8)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('SOC Error (%)')
    axes[1].set_title(f'SOC Estimation Error (RMSE={rmse:.4f}%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    # NIS and robust weight
    ax2_twin = axes[2].twinx()
    axes[2].plot(time, results['NIS'], 'b-', linewidth=0.5, alpha=0.7, label='NIS')
    axes[2].axhline(y=6.63, color='r', linestyle='--', linewidth=1, label='NIS Threshold')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('NIS', color='b')
    axes[2].set_ylim([0, min(20, np.max(results['NIS']) * 1.1)])
    axes[2].legend(loc='upper left')

    ax2_twin.plot(time, results['robust_weight'], 'g-', linewidth=0.5, alpha=0.7)
    ax2_twin.set_ylabel('Robust Weight', color='g')
    ax2_twin.set_ylim([0, 1.1])

    axes[2].set_title('NIS and Robust Weight')
    axes[2].grid(True, alpha=0.3)

    # Adaptive nu
    axes[3].plot(time, results['nu'], 'm-', linewidth=0.8)
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Student-t ν')
    axes[3].set_title('Adaptive Degrees of Freedom')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "RobustSRUKF_results.png"
    plt.savefig(save_path, dpi=150)
    print(f"Results saved to: {save_path}")
    plt.close()

    return results, {'rmse': rmse, 'mae': mae, 'max_error': max_error, 'nis_triggered': nis_triggered}


def compare_algorithms(data_path: Path, save_dir: Path):
    """Compare AEKF, UKF and Robust SR-UKF"""
    print("\n" + "="*50)
    print("Comparing AEKF vs UKF vs Robust SR-UKF")
    print("="*50)

    df = pd.read_csv(data_path)
    voltage = df['voltage_V'].values
    current = df['current_A'].values
    soc_true = df['soc_percent'].values
    time = df['time_s'].values

    initial_soc = soc_true[0] / 100

    # AEKF
    aekf = AEKF(initial_soc=initial_soc, capacity_Ah=2.0, use_online_param_id=True)
    aekf_results = aekf.estimate_batch(voltage, current, soc_true, initial_soc)

    # UKF
    ukf = UKF(initial_soc=initial_soc, capacity_Ah=2.0, use_online_param_id=True)
    ukf_results = ukf.estimate_batch(voltage, current, soc_true, initial_soc)

    # Robust SR-UKF
    sr_ukf = RobustSRUKF(initial_soc=initial_soc, capacity_Ah=2.0, use_online_param_id=True)
    sr_ukf_results = sr_ukf.estimate_batch(voltage, current, soc_true, initial_soc)

    # Calculate errors
    aekf_error = aekf_results['SOC_percent'] - soc_true
    ukf_error = ukf_results['SOC_percent'] - soc_true
    sr_ukf_error = sr_ukf_results['SOC_percent'] - soc_true

    aekf_rmse = np.sqrt(np.mean(aekf_error**2))
    ukf_rmse = np.sqrt(np.mean(ukf_error**2))
    sr_ukf_rmse = np.sqrt(np.mean(sr_ukf_error**2))

    print(f"\nComparison:")
    print(f"  AEKF RMSE:      {aekf_rmse:.4f}%")
    print(f"  UKF RMSE:       {ukf_rmse:.4f}%")
    print(f"  SR-UKF RMSE:    {sr_ukf_rmse:.4f}%")

    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(time, soc_true, 'b-', linewidth=1.5, label='True SOC')
    axes[0].plot(time, aekf_results['SOC_percent'], 'r--', linewidth=1, label=f'AEKF (RMSE={aekf_rmse:.3f}%)')
    axes[0].plot(time, ukf_results['SOC_percent'], 'g-.', linewidth=1, label=f'UKF (RMSE={ukf_rmse:.3f}%)')
    axes[0].plot(time, sr_ukf_results['SOC_percent'], 'm:', linewidth=1.5, label=f'SR-UKF (RMSE={sr_ukf_rmse:.3f}%)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('SOC (%)')
    axes[0].set_title('SOC Estimation Comparison: AEKF vs UKF vs Robust SR-UKF')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, aekf_error, 'r-', linewidth=0.8, alpha=0.7, label='AEKF Error')
    axes[1].plot(time, ukf_error, 'g-', linewidth=0.8, alpha=0.7, label='UKF Error')
    axes[1].plot(time, sr_ukf_error, 'm-', linewidth=0.8, alpha=0.7, label='SR-UKF Error')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('SOC Error (%)')
    axes[1].set_title('Estimation Error Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    save_path = save_dir / "All_algorithms_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"Comparison saved to: {save_path}")
    plt.close()


def main():
    """Main function"""
    # Data path
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / "dataset" / "processed" / "25C_DST_80SOC.csv"
    save_dir = base_path / "dataset" / "processed"

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("Please run process_battery_data.py first.")
        return

    print(f"Loading data from: {data_path}")

    # Test AEKF
    aekf_results, aekf_metrics = test_aekf(data_path, save_dir)

    # Test UKF
    ukf_results, ukf_metrics = test_ukf(data_path, save_dir)

    # Test Robust SR-UKF
    sr_ukf_results, sr_ukf_metrics = test_sr_ukf(data_path, save_dir)

    # Compare all algorithms
    compare_algorithms(data_path, save_dir)

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"{'Algorithm':<12} {'RMSE':<12} {'MAE':<12} {'Max Error':<12}")
    print("-"*60)
    print(f"{'AEKF':<12} {aekf_metrics['rmse']:.4f}%{'':<6} {aekf_metrics['mae']:.4f}%{'':<6} {aekf_metrics['max_error']:.4f}%")
    print(f"{'UKF':<12} {ukf_metrics['rmse']:.4f}%{'':<6} {ukf_metrics['mae']:.4f}%{'':<6} {ukf_metrics['max_error']:.4f}%")
    print(f"{'SR-UKF':<12} {sr_ukf_metrics['rmse']:.4f}%{'':<6} {sr_ukf_metrics['mae']:.4f}%{'':<6} {sr_ukf_metrics['max_error']:.4f}%")
    print("-"*60)
    print(f"SR-UKF NIS Gate Triggered: {sr_ukf_metrics['nis_triggered']} times")


if __name__ == "__main__":
    main()
