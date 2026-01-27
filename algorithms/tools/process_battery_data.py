"""
Battery Test Data Processing Tool

This script processes battery test data from various conditions and formats it
into standardized CSV files for further analysis.

Output CSV format:
    - time_s: Time in seconds (relative to test start)
    - voltage_V: Terminal voltage in Volts
    - current_A: Current in Amperes (positive=charge, negative=discharge)
    - soc_percent: State of Charge in percentage (0-100)

Naming convention:
    {temperature}C_{profile}_{initial_soc}SOC.csv
    Example: 25C_DST_80SOC.csv

Author: Auto-generated
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import argparse


class BatteryDataProcessor:
    """Process battery test data into standardized CSV format."""

    # Supported test profiles and their folder names
    PROFILE_MAPPING = {
        'DST': 'DST',
        'BBDST': 'BBDST',
        'BJDST': 'BBDST',  # Alias (used in 0C and 45C data)
        'FUDS': 'FUDS',
        'US06': 'US06',
    }

    # Reverse mapping for output filename normalization
    OUTPUT_PROFILE_NAME = {
        'DST': 'DST',
        'BBDST': 'BBDST',
        'BJDST': 'BBDST',  # Normalize BJDST to BBDST in output
        'FUDS': 'FUDS',
        'US06': 'US06',
    }

    def __init__(self, base_path: str = None, rated_capacity: float = 2.0):
        """
        Initialize the data processor.

        Args:
            base_path: Base path to the dataset folder
            rated_capacity: Battery rated capacity in Ah (default: 2.0 for 2000mAh)
        """
        if base_path is None:
            self.base_path = Path(__file__).parent.parent.parent / "dataset"
        else:
            self.base_path = Path(base_path)

        self.rated_capacity = rated_capacity
        self.output_dir = self.base_path / "processed"

    def find_data_file(self, temperature: int, profile: str, initial_soc: int) -> Optional[Path]:
        """
        Find the data file matching the specified parameters.

        Args:
            temperature: Temperature in Celsius
            profile: Test profile (DST, BBDST, FUDS, US06)
            initial_soc: Initial SOC percentage

        Returns:
            Path to the data file, or None if not found
        """
        # Normalize profile name
        profile_upper = profile.upper()
        if profile_upper in self.PROFILE_MAPPING:
            folder_name = self.PROFILE_MAPPING[profile_upper]
        else:
            folder_name = profile_upper

        # 尝试的文件夹名列表 (BBDST和BJDST互为别名)
        folder_candidates = [folder_name]
        if folder_name == 'BBDST':
            folder_candidates.append('BJDST')
        elif folder_name == 'BJDST':
            folder_candidates.append('BBDST')

        # Try each candidate folder
        for candidate in folder_candidates:
            search_path = self.base_path / f"{temperature}C" / candidate

            if not search_path.exists():
                continue

            # Search for matching file
            for file in search_path.glob("*.xls*"):
                # Check if file name contains the SOC value
                if f"{initial_soc}SOC" in file.name:
                    return file

        return None

    def load_and_process_data(
        self,
        temperature: int,
        profile: str,
        initial_soc: int,
        step_index: int = 7
    ) -> Optional[pd.DataFrame]:
        """
        Load and process battery test data.

        Args:
            temperature: Temperature in Celsius
            profile: Test profile name
            initial_soc: Initial SOC percentage
            step_index: Step index for dynamic test data (default: 7)

        Returns:
            Processed DataFrame or None if file not found
        """
        # Find data file
        file_path = self.find_data_file(temperature, profile, initial_soc)

        if file_path is None:
            print(f"Error: No data file found for {temperature}C_{profile}_{initial_soc}SOC")
            return None

        print(f"Processing: {file_path.name}")

        # Load Excel file
        xlsx = pd.ExcelFile(file_path)

        # Find channel data sheet
        data_sheet = None
        for sheet in xlsx.sheet_names:
            if sheet.startswith("Channel"):
                data_sheet = sheet
                break

        if data_sheet is None:
            print(f"Error: No Channel data sheet found in {file_path}")
            return None

        # Read data
        df = pd.read_excel(xlsx, sheet_name=data_sheet)

        # Extract dynamic test data (specified step)
        test_data = df[df['Step_Index'] == step_index].copy().reset_index(drop=True)

        if len(test_data) == 0:
            print(f"Warning: No data found for Step_Index={step_index}")
            # Try to find the main test step
            step_counts = df.groupby('Step_Index').size()
            main_step = step_counts.idxmax()
            print(f"Using Step_Index={main_step} instead (most data points)")
            test_data = df[df['Step_Index'] == main_step].copy().reset_index(drop=True)

        # Calculate time (uniform 1s sampling)
        # Original data is approximately 1s interval, so we use integer seconds directly
        n_samples = len(test_data)
        time_s = np.arange(n_samples, dtype=float)  # 0, 1, 2, 3, ... seconds

        # Get voltage
        voltage_V = test_data['Voltage(V)'].values

        # Get current (already in correct convention: positive=charge, negative=discharge)
        current_A = test_data['Current(A)'].values

        # Calculate SOC using cumulative capacity method
        charge_cap_start = test_data['Charge_Capacity(Ah)'].iloc[0]
        discharge_cap_start = test_data['Discharge_Capacity(Ah)'].iloc[0]

        delta_charge = test_data['Charge_Capacity(Ah)'].values - charge_cap_start
        delta_discharge = test_data['Discharge_Capacity(Ah)'].values - discharge_cap_start

        # SOC = initial_SOC + (charge - discharge) / rated_capacity * 100
        soc_percent = initial_soc + (delta_charge - delta_discharge) / self.rated_capacity * 100

        # Clip SOC to valid range [0, 100]
        soc_percent = np.clip(soc_percent, 0, 100)

        # Create output DataFrame
        result_df = pd.DataFrame({
            'time_s': time_s,
            'voltage_V': voltage_V,
            'current_A': current_A,
            'soc_percent': soc_percent
        })

        # Round values for cleaner output
        result_df['time_s'] = result_df['time_s'].round(3)
        result_df['voltage_V'] = result_df['voltage_V'].round(6)
        result_df['current_A'] = result_df['current_A'].round(6)
        result_df['soc_percent'] = result_df['soc_percent'].round(4)

        return result_df

    def save_processed_data(
        self,
        df: pd.DataFrame,
        temperature: int,
        profile: str,
        initial_soc: int,
        output_dir: Path = None
    ) -> Path:
        """
        Save processed data to CSV file.

        Args:
            df: Processed DataFrame
            temperature: Temperature in Celsius
            profile: Test profile name
            initial_soc: Initial SOC percentage
            output_dir: Output directory (default: dataset/processed/{temperature}C)

        Returns:
            Path to saved file
        """
        if output_dir is None:
            # 默认保存到 processed/{temperature}C/ 子文件夹
            output_dir = self.output_dir / f"{temperature}C"

        # Create output directory if not exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename (不再包含温度前缀，因为已经在子文件夹中)
        # 使用 OUTPUT_PROFILE_NAME 来规范化配置文件名称 (BJDST -> BBDST)
        normalized_profile = self.OUTPUT_PROFILE_NAME.get(profile.upper(), profile.upper())
        filename = f"{normalized_profile}_{initial_soc}SOC.csv"
        output_path = output_dir / filename

        # Save to CSV
        df.to_csv(output_path, index=False)

        return output_path

    def process_single(
        self,
        temperature: int,
        profile: str,
        initial_soc: int,
        output_dir: Path = None
    ) -> Optional[Path]:
        """
        Process a single test data file.

        Args:
            temperature: Temperature in Celsius
            profile: Test profile name
            initial_soc: Initial SOC percentage
            output_dir: Output directory

        Returns:
            Path to saved file, or None if processing failed
        """
        # Load and process data
        df = self.load_and_process_data(temperature, profile, initial_soc)

        if df is None:
            return None

        # Save to CSV
        output_path = self.save_processed_data(df, temperature, profile, initial_soc, output_dir)

        # Print summary
        print(f"  Output: {output_path.name}")
        print(f"  Records: {len(df)}")
        print(f"  Duration: {df['time_s'].max():.1f} s ({df['time_s'].max()/60:.1f} min)")
        print(f"  Voltage: {df['voltage_V'].min():.3f} V ~ {df['voltage_V'].max():.3f} V")
        print(f"  Current: {df['current_A'].min():.3f} A ~ {df['current_A'].max():.3f} A")
        print(f"  SOC: {df['soc_percent'].min():.1f}% ~ {df['soc_percent'].max():.1f}%")
        print()

        return output_path

    def process_all(self, output_dir: Path = None) -> List[Path]:
        """
        Process all available data files.

        Args:
            output_dir: Output directory

        Returns:
            List of paths to saved files
        """
        saved_files = []

        # Find all temperature folders
        for temp_folder in self.base_path.iterdir():
            if not temp_folder.is_dir():
                continue
            if not temp_folder.name.endswith('C'):
                continue

            # Extract temperature
            try:
                temperature = int(temp_folder.name[:-1])
            except ValueError:
                continue

            # Find all profile folders
            for profile_folder in temp_folder.iterdir():
                if not profile_folder.is_dir():
                    continue

                profile = profile_folder.name

                # Find all data files
                for data_file in profile_folder.glob("*.xls*"):
                    # Extract initial SOC from filename
                    filename = data_file.name
                    soc_match = None

                    # Look for SOC in filename (e.g., "50SOC" or "80SOC")
                    for soc in [50, 80, 100, 30, 60, 70, 90, 40, 20, 10]:
                        if f"{soc}SOC" in filename:
                            soc_match = soc
                            break

                    if soc_match is None:
                        print(f"Warning: Could not extract SOC from {filename}, skipping")
                        continue

                    # Process this file
                    result = self.process_single(temperature, profile, soc_match, output_dir)
                    if result:
                        saved_files.append(result)

        return saved_files

    def list_available_data(self) -> List[Tuple[int, str, int]]:
        """
        List all available data files.

        Returns:
            List of tuples (temperature, profile, initial_soc)
        """
        available = []

        for temp_folder in self.base_path.iterdir():
            if not temp_folder.is_dir() or not temp_folder.name.endswith('C'):
                continue

            try:
                temperature = int(temp_folder.name[:-1])
            except ValueError:
                continue

            for profile_folder in temp_folder.iterdir():
                if not profile_folder.is_dir():
                    continue

                profile = profile_folder.name

                for data_file in profile_folder.glob("*.xls*"):
                    filename = data_file.name
                    for soc in [50, 80, 100, 30, 60, 70, 90, 40, 20, 10]:
                        if f"{soc}SOC" in filename:
                            available.append((temperature, profile, soc))
                            break

        return sorted(available)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Process battery test data into standardized CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python process_battery_data.py -t 25 -p DST -s 80

  # Process all available data
  python process_battery_data.py --all

  # List available data
  python process_battery_data.py --list

  # Specify output directory
  python process_battery_data.py -t 25 -p DST -s 80 -o ./output

Output CSV format:
  - time_s: Time in seconds
  - voltage_V: Terminal voltage (V)
  - current_A: Current (A), positive=charge, negative=discharge
  - soc_percent: State of Charge (%)

Naming convention:
  {temperature}C_{profile}_{initial_soc}SOC.csv
  Example: 25C_DST_80SOC.csv
        """
    )

    parser.add_argument('-t', '--temperature', type=int, help='Temperature in Celsius')
    parser.add_argument('-p', '--profile', type=str, help='Test profile (DST, BBDST, FUDS, US06)')
    parser.add_argument('-s', '--soc', type=int, help='Initial SOC percentage')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output directory')
    parser.add_argument('-b', '--base-path', type=str, default=None, help='Base dataset path')
    parser.add_argument('-c', '--capacity', type=float, default=2.0, help='Rated capacity in Ah')
    parser.add_argument('--all', action='store_true', help='Process all available data')
    parser.add_argument('--list', action='store_true', help='List available data files')

    args = parser.parse_args()

    # Create processor
    processor = BatteryDataProcessor(base_path=args.base_path, rated_capacity=args.capacity)

    # Handle --list
    if args.list:
        print("Available data files:")
        print("-" * 40)
        available = processor.list_available_data()
        for temp, profile, soc in available:
            print(f"  {temp}C  {profile:8s}  {soc}% SOC")
        print("-" * 40)
        print(f"Total: {len(available)} files")
        return

    # Handle --all
    if args.all:
        print("Processing all available data files...")
        print("=" * 50)
        output_dir = Path(args.output) if args.output else None
        saved_files = processor.process_all(output_dir)
        print("=" * 50)
        print(f"Processed {len(saved_files)} files")
        return

    # Process single file
    if args.temperature is None or args.profile is None or args.soc is None:
        parser.print_help()
        print("\nError: Must specify -t, -p, -s or use --all/--list")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else None
    result = processor.process_single(args.temperature, args.profile, args.soc, output_dir)

    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
