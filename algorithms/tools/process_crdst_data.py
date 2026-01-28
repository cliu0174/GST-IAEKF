"""
CRDST Data Processing Tool

This script processes CRDST battery test data from .mat files and converts them
into standardized CSV format compatible with the existing CALCE dataset.

Input .mat files:
    - lidj1DST.mat: Current data (A)
    - ludj1DST.mat: Voltage data (V)
    - socdj1DST.mat: SOC data (0-1 scale, with multiple fields)

Output CSV format (same as other datasets):
    - time_s: Time in seconds (1s interval)
    - voltage_V: Terminal voltage in Volts
    - current_A: Current in Amperes (positive=charge, negative=discharge)
    - soc_percent: State of Charge in percentage (0-100)

Naming convention:
    {profile}_{initial_soc}SOC.csv
    Example: CRDST_80SOC.csv

Author: Auto-generated
Date: 2024
"""

import pandas as pd
import scipy.io as sio
import numpy as np
from pathlib import Path
import argparse
import sys


class CRDSTDataProcessor:
    """Process CRDST .mat files into standardized CSV format."""

    def __init__(self, base_path: str = None):
        """
        Initialize the CRDST data processor.

        Args:
            base_path: Base path to the CRDST dataset folder
        """
        if base_path is None:
            self.base_path = Path(__file__).parent.parent.parent / "dataset" / "CRDST"
        else:
            self.base_path = Path(base_path)

        self.output_dir = Path(__file__).parent.parent.parent / "dataset" / "processed"

    def process_crdst_data(
        self,
        temperature: int,
        profile: str = 'CRDST',
        current_file: str = 'lidj1DST.mat',
        voltage_file: str = 'ludj1DST.mat',
        soc_file: str = 'socdj1DST.mat',
        soc_field: str = 'SOC_Estimated',
        output_dir: Path = None
    ) -> tuple:
        """
        Process CRDST .mat files and generate standardized CSV.

        Args:
            temperature: Temperature in Celsius
            profile: Test profile name (default: 'CRDST')
            current_file: Current .mat filename
            voltage_file: Voltage .mat filename
            soc_file: SOC .mat filename
            soc_field: Field name in SOC .mat file to use (default: 'SOC_Estimated')
            output_dir: Output directory (default: dataset/processed/{temperature}C)

        Returns:
            tuple: (output_path, dataframe)
        """
        # Construct paths
        mat_dir = self.base_path / f"{temperature}C"
        if not mat_dir.exists():
            raise FileNotFoundError(f"Directory not found: {mat_dir}")

        current_path = mat_dir / current_file
        voltage_path = mat_dir / voltage_file
        soc_path = mat_dir / soc_file

        # Check files exist
        for path, name in [(current_path, 'Current'), (voltage_path, 'Voltage'), (soc_path, 'SOC')]:
            if not path.exists():
                raise FileNotFoundError(f"{name} file not found: {path}")

        # Read .mat files
        print(f"Reading .mat files from {mat_dir}...")
        current_mat = sio.loadmat(current_path)
        voltage_mat = sio.loadmat(voltage_path)
        soc_mat = sio.loadmat(soc_path)

        # Extract data arrays
        # Get the first non-metadata key
        current_key = [k for k in current_mat.keys() if not k.startswith('__')][0]
        voltage_key = [k for k in voltage_mat.keys() if not k.startswith('__')][0]

        current_A = current_mat[current_key].flatten()
        voltage_V = voltage_mat[voltage_key].flatten()

        # SOC data - use specified field or default to SOC_Estimated
        if soc_field in soc_mat:
            soc_data = soc_mat[soc_field].flatten()
        else:
            # Try common field names
            for field in ['SOC_Estimated', 'SOC_Compensated', 'SOC']:
                if field in soc_mat:
                    soc_field = field
                    soc_data = soc_mat[field].flatten()
                    print(f"Using SOC field: {soc_field}")
                    break
            else:
                available_fields = [k for k in soc_mat.keys() if not k.startswith('__')]
                raise ValueError(f"SOC field '{soc_field}' not found. Available fields: {available_fields}")

        print(f"\nData shapes:")
        print(f"  Current: {current_A.shape}")
        print(f"  Voltage: {voltage_V.shape}")
        print(f"  SOC: {soc_data.shape}")

        # Handle length mismatch (SOC may be 1 sample shorter)
        min_length = min(len(current_A), len(voltage_V), len(soc_data))
        if len(current_A) != len(soc_data) or len(voltage_V) != len(soc_data):
            print(f"\nWarning: Data length mismatch detected. Trimming to {min_length} samples.")
            current_A = current_A[:min_length]
            voltage_V = voltage_V[:min_length]
            soc_data = soc_data[:min_length]

        # Convert SOC to percentage (0-100)
        # Check if data is already in percentage or needs conversion
        if soc_data.max() <= 1.0:
            soc_percent = soc_data * 100
            print(f"Converting SOC from 0-1 scale to percentage")
        else:
            soc_percent = soc_data
            print(f"SOC data already in percentage scale")

        # Generate time series (1s interval)
        n_points = len(soc_percent)
        time_s = np.arange(n_points, dtype=float)

        # Create DataFrame
        df = pd.DataFrame({
            'time_s': time_s,
            'voltage_V': voltage_V,
            'current_A': current_A,
            'soc_percent': soc_percent
        })

        # Round values for consistency with other datasets
        df['time_s'] = df['time_s'].round(3)
        df['voltage_V'] = df['voltage_V'].round(6)
        df['current_A'] = df['current_A'].round(6)
        df['soc_percent'] = df['soc_percent'].round(4)

        # Determine output filename based on initial SOC
        actual_initial_soc = int(round(soc_percent[0]))
        print(f"\nActual initial SOC: {actual_initial_soc}%")

        # Map to standard SOC labels (50 or 80)
        if actual_initial_soc >= 65:
            soc_label = 80
        else:
            soc_label = 50

        # Set output directory
        if output_dir is None:
            output_dir = self.output_dir / f"{temperature}C"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"{profile}_{soc_label}SOC.csv"
        output_path = output_dir / filename

        # Save CSV
        df.to_csv(output_path, index=False)

        # Print summary
        print(f"\nProcessing complete!")
        print(f"  Output: {output_path}")
        print(f"  Records: {len(df)}")
        print(f"  Duration: {df['time_s'].max():.1f} s ({df['time_s'].max()/60:.1f} min)")
        print(f"  Voltage: {df['voltage_V'].min():.3f} V ~ {df['voltage_V'].max():.3f} V")
        print(f"  Current: {df['current_A'].min():.3f} A ~ {df['current_A'].max():.3f} A")
        print(f"  SOC: {df['soc_percent'].min():.1f}% ~ {df['soc_percent'].max():.1f}%")

        # Display sample data
        print(f"\nFirst 5 rows:")
        print(df.head())

        print(f"\nLast 5 rows:")
        print(df.tail())

        return output_path, df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Process CRDST .mat files into standardized CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process CRDST data at 25C
  python process_crdst_data.py -t 25

  # Process with custom profile name
  python process_crdst_data.py -t 25 -p CRDST

  # Specify custom file names
  python process_crdst_data.py -t 25 --current lidj1DST.mat --voltage ludj1DST.mat --soc socdj1DST.mat

  # Specify output directory
  python process_crdst_data.py -t 25 -o ./output

Output CSV format:
  - time_s: Time in seconds (1s interval)
  - voltage_V: Terminal voltage (V)
  - current_A: Current (A), positive=charge, negative=discharge
  - soc_percent: State of Charge (%)

Naming convention:
  {profile}_{initial_soc}SOC.csv
  Example: CRDST_80SOC.csv
        """
    )

    parser.add_argument('-t', '--temperature', type=int, required=True,
                        help='Temperature in Celsius (e.g., 25)')
    parser.add_argument('-p', '--profile', type=str, default='CRDST',
                        help='Test profile name (default: CRDST)')
    parser.add_argument('--current', type=str, default='lidj1DST.mat',
                        help='Current .mat filename (default: lidj1DST.mat)')
    parser.add_argument('--voltage', type=str, default='ludj1DST.mat',
                        help='Voltage .mat filename (default: ludj1DST.mat)')
    parser.add_argument('--soc', type=str, default='socdj1DST.mat',
                        help='SOC .mat filename (default: socdj1DST.mat)')
    parser.add_argument('--soc-field', type=str, default='SOC_Estimated',
                        help='SOC field name in .mat file (default: SOC_Estimated)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory (default: dataset/processed/{temperature}C)')
    parser.add_argument('-b', '--base-path', type=str, default=None,
                        help='Base CRDST dataset path (default: dataset/CRDST)')

    args = parser.parse_args()

    # Create processor
    processor = CRDSTDataProcessor(base_path=args.base_path)

    # Process data
    try:
        output_dir = Path(args.output) if args.output else None
        output_path, df = processor.process_crdst_data(
            temperature=args.temperature,
            profile=args.profile,
            current_file=args.current,
            voltage_file=args.voltage,
            soc_file=args.soc,
            soc_field=args.soc_field,
            output_dir=output_dir
        )
        print(f"\nSuccess! File saved to: {output_path}")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
