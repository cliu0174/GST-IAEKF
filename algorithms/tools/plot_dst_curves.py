"""
DST (Dynamic Stress Test) Data Visualization Tool

This script plots SOC, Voltage, and Current curves for DST test data.
Supports different temperatures and initial SOC levels.

Author: Auto-generated
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class DSTDataPlotter:
    """Class for loading and plotting DST test data."""

    def __init__(self, base_path: str = None):
        """
        Initialize the DST data plotter.

        Args:
            base_path: Base path to the CALCE dataset. If None, uses default path.
        """
        if base_path is None:
            # Default path - adjust as needed
            self.base_path = Path(__file__).parent.parent.parent / "dataset"
        else:
            self.base_path = Path(base_path)

        # Battery parameters
        self.rated_capacity = 2.0  # Ah (2000mAh)

    def get_dst_file_path(self, temperature: int, initial_soc: int) -> Path:
        """
        Get the file path for DST data.

        Args:
            temperature: Temperature in Celsius (e.g., 25)
            initial_soc: Initial SOC percentage (e.g., 50 or 80)

        Returns:
            Path to the DST data file
        """
        temp_folder = f"{temperature}C"

        # Search for matching DST file
        dst_folder = self.base_path / temp_folder / "DST"

        if not dst_folder.exists():
            raise FileNotFoundError(f"DST folder not found: {dst_folder}")

        # Find file matching the SOC
        for file in dst_folder.glob(f"*DST*{initial_soc}SOC*.xls*"):
            return file

        raise FileNotFoundError(
            f"No DST file found for {temperature}C and {initial_soc}% SOC"
        )

    def load_dst_data(self, temperature: int, initial_soc: int) -> pd.DataFrame:
        """
        Load DST test data from Excel file.

        Args:
            temperature: Temperature in Celsius
            initial_soc: Initial SOC percentage

        Returns:
            DataFrame containing the test data
        """
        file_path = self.get_dst_file_path(temperature, initial_soc)
        print(f"Loading data from: {file_path}")

        # Read Excel file - find the data sheet
        xlsx = pd.ExcelFile(file_path)

        # Find the channel data sheet
        data_sheet = None
        for sheet in xlsx.sheet_names:
            if sheet.startswith("Channel"):
                data_sheet = sheet
                break

        if data_sheet is None:
            raise ValueError(f"No Channel data sheet found in {file_path}")

        df = pd.read_excel(xlsx, sheet_name=data_sheet)
        return df

    def calculate_soc(self, df: pd.DataFrame, initial_soc: float) -> np.ndarray:
        """
        Calculate SOC from capacity data.

        Args:
            df: DataFrame with test data
            initial_soc: Initial SOC (0-1 scale)

        Returns:
            Array of SOC values (0-100 scale)
        """
        # Net capacity change = Charge - Discharge
        charge_cap = df['Charge_Capacity(Ah)'].values
        discharge_cap = df['Discharge_Capacity(Ah)'].values

        # SOC = initial_SOC + (charge - discharge) / rated_capacity
        net_capacity = charge_cap - discharge_cap
        soc = (initial_soc + net_capacity / self.rated_capacity) * 100

        return soc

    def plot_dst_curves(
        self,
        temperature: int,
        initial_soc_list: list = [50, 80],
        save_path: str = None,
        show_plot: bool = True
    ):
        """
        Plot SOC, Voltage, and Current curves for DST test.

        Args:
            temperature: Temperature in Celsius
            initial_soc_list: List of initial SOC values to plot
            save_path: Path to save the figure. If None, saves to default location.
            show_plot: Whether to display the plot
        """
        n_tests = len(initial_soc_list)

        # Create figure with subplots
        fig, axes = plt.subplots(3, n_tests, figsize=(7 * n_tests, 12))

        # Ensure axes is 2D
        if n_tests == 1:
            axes = axes.reshape(-1, 1)

        colors = {'current': '#1f77b4', 'voltage': '#d62728', 'soc': '#2ca02c'}

        for col, initial_soc in enumerate(initial_soc_list):
            try:
                # Load data
                df = self.load_dst_data(temperature, initial_soc)

                # Extract DST cycle data (Step 7)
                dst_data = df[df['Step_Index'] == 7].copy().reset_index(drop=True)

                if len(dst_data) == 0:
                    print(f"Warning: No DST cycle data (Step 7) found for {initial_soc}% SOC")
                    continue

                # Time in seconds (use Test_Time for continuous tracking)
                # Calculate relative time from DST start
                start_time = dst_data['Test_Time(s)'].iloc[0]
                time_s = dst_data['Test_Time(s)'].values - start_time

                # Current
                current = dst_data['Current(A)'].values

                # Voltage
                voltage = dst_data['Voltage(V)'].values

                # Calculate SOC using cumulative capacity
                # Get capacity at DST start
                charge_cap_start = dst_data['Charge_Capacity(Ah)'].iloc[0]
                discharge_cap_start = dst_data['Discharge_Capacity(Ah)'].iloc[0]

                # Calculate delta capacity from DST start
                delta_charge = dst_data['Charge_Capacity(Ah)'].values - charge_cap_start
                delta_discharge = dst_data['Discharge_Capacity(Ah)'].values - discharge_cap_start

                # SOC = initial_SOC + (delta_charge - delta_discharge) / rated_capacity * 100
                soc = initial_soc + (delta_charge - delta_discharge) / self.rated_capacity * 100

                # Plot Current
                ax_current = axes[0, col]
                ax_current.plot(time_s, current, color=colors['current'], linewidth=0.8)
                ax_current.set_xlabel('Time (s)', fontsize=11)
                ax_current.set_ylabel('Current (A)', fontsize=11)
                ax_current.set_title(f'{temperature}°C DST - {initial_soc}% Initial SOC\nCurrent Profile', fontsize=12)
                ax_current.grid(True, alpha=0.3)
                ax_current.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
                ax_current.set_xlim(0, time_s.max())

                # Plot Voltage
                ax_voltage = axes[1, col]
                ax_voltage.plot(time_s, voltage, color=colors['voltage'], linewidth=0.8)
                ax_voltage.set_xlabel('Time (s)', fontsize=11)
                ax_voltage.set_ylabel('Voltage (V)', fontsize=11)
                ax_voltage.set_title(f'Voltage Response', fontsize=12)
                ax_voltage.grid(True, alpha=0.3)
                ax_voltage.set_xlim(0, time_s.max())

                # Plot SOC
                ax_soc = axes[2, col]
                ax_soc.plot(time_s, soc, color=colors['soc'], linewidth=0.8)
                ax_soc.set_xlabel('Time (s)', fontsize=11)
                ax_soc.set_ylabel('SOC (%)', fontsize=11)
                ax_soc.set_title(f'State of Charge', fontsize=12)
                ax_soc.grid(True, alpha=0.3)
                ax_soc.set_xlim(0, time_s.max())

                # Print statistics
                print(f"\n{temperature}°C DST - {initial_soc}% Initial SOC:")
                print(f"  Duration: {time_s.max():.1f} s")
                print(f"  Current range: {current.min():.3f} A to {current.max():.3f} A")
                print(f"  Voltage range: {voltage.min():.3f} V to {voltage.max():.3f} V")
                print(f"  SOC range: {soc.min():.1f}% to {soc.max():.1f}%")

            except FileNotFoundError as e:
                print(f"Error: {e}")
                continue

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = self.base_path / f"{temperature}C" / "DST" / f"DST_{temperature}C_curves.png"

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig


def main():
    """Main function to run the DST plotting tool."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot DST (Dynamic Stress Test) curves for battery data'
    )
    parser.add_argument(
        '-t', '--temperature',
        type=int,
        default=25,
        help='Temperature in Celsius (default: 25)'
    )
    parser.add_argument(
        '-s', '--soc',
        type=int,
        nargs='+',
        default=[50, 80],
        help='Initial SOC values to plot (default: 50 80)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file path for the figure'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display the plot (only save)'
    )
    parser.add_argument(
        '-b', '--base-path',
        type=str,
        default=None,
        help='Base path to the dataset folder'
    )

    args = parser.parse_args()

    # Create plotter and generate plots
    plotter = DSTDataPlotter(base_path=args.base_path)

    try:
        plotter.plot_dst_curves(
            temperature=args.temperature,
            initial_soc_list=args.soc,
            save_path=args.output,
            show_plot=not args.no_show
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nAvailable temperature folders:")
        for folder in plotter.base_path.iterdir():
            if folder.is_dir() and folder.name.endswith('C'):
                print(f"  - {folder.name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
