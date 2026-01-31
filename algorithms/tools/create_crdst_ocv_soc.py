"""
CRDST OCV-SOC Curve Generator

This script generates OCV-SOC curve data for the CRDST battery dataset.
The data is based on the original MATLAB lookup table and polynomial fitting.

Original MATLAB data:
    ocv_lut = [2.789261, 3.20478, 3.242465, 3.275661, 3.289715, 3.290776,
               3.293613, 3.329685, 3.330853, 3.33225, 3.380374]
    soc_lut = 0:0.1:1  (0 to 1 with 0.1 step)
    pfit = polyfit(soc_lut, ocv_lut, 7)  (7th order polynomial)

Output files:
    - Discharge_OCV_SOC_data.csv: Original lookup table data
    - polynomial_coefficients.csv: 7th order polynomial coefficients
    - polynomial_fit.png: Visualization of OCV-SOC curve

Author: Generated
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


class CRDSTOCVGenerator:
    """Generate OCV-SOC curve data for CRDST battery."""

    # Original MATLAB lookup table data
    OCV_LUT = np.array([2.789261, 3.20478, 3.242465, 3.275661, 3.289715,
                        3.290776, 3.293613, 3.329685, 3.330853, 3.33225, 3.380374])
    SOC_LUT = np.arange(0, 1.1, 0.1)  # 0 to 1 with 0.1 step

    def __init__(self, output_dir: str = None):
        """
        Initialize the OCV-SOC generator.

        Args:
            output_dir: Output directory for OCV-SOC data files
        """
        if output_dir is None:
            self.output_dir = Path(__file__).parent.parent.parent / "dataset" / "processed" / "CRDST" / "OCV-SOC" / "Sample1"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fit_polynomial(self, order: int = 7) -> np.ndarray:
        """
        Fit polynomial to OCV-SOC data.

        Args:
            order: Polynomial order (default: 7, matching MATLAB code)

        Returns:
            Polynomial coefficients (highest order first)
        """
        # Fit polynomial using numpy (same as MATLAB polyfit)
        coeffs = np.polyfit(self.SOC_LUT, self.OCV_LUT, order)
        return coeffs

    def evaluate_polynomial(self, coeffs: np.ndarray, soc_values: np.ndarray) -> np.ndarray:
        """
        Evaluate polynomial at given SOC values.

        Args:
            coeffs: Polynomial coefficients
            soc_values: SOC values (0-1 range)

        Returns:
            OCV values
        """
        return np.polyval(coeffs, soc_values)

    def save_original_data(self) -> Path:
        """
        Save original lookup table data to CSV.

        Returns:
            Path to saved file
        """
        # Create DataFrame
        df = pd.DataFrame({
            'SOC': self.SOC_LUT,
            'OCV(V)': self.OCV_LUT
        })

        # Round values
        df['SOC'] = df['SOC'].round(6)
        df['OCV(V)'] = df['OCV(V)'].round(6)

        # Save to CSV
        output_path = self.output_dir / "Discharge_OCV_SOC_data.csv"
        df.to_csv(output_path, index=False)

        print(f"Original OCV-SOC data saved to: {output_path}")
        return output_path

    def save_polynomial_coefficients(self, coeffs: np.ndarray, order: int = 7) -> Path:
        """
        Save polynomial coefficients to CSV.

        Args:
            coeffs: Polynomial coefficients
            order: Polynomial order

        Returns:
            Path to saved file
        """
        # Create coefficient labels (p0 = highest order, p7 = constant)
        labels = [f'p{i}' for i in range(len(coeffs))]

        # Create DataFrame
        df = pd.DataFrame({
            'Coefficient': labels,
            'Value': coeffs
        })

        # Add polynomial formula as comment
        formula = "OCV(SOC) = "
        terms = []
        for i, coeff in enumerate(coeffs):
            power = order - i
            if power > 1:
                terms.append(f"p{i}*SOC^{power}")
            elif power == 1:
                terms.append(f"p{i}*SOC")
            else:
                terms.append(f"p{i}")
        formula += " + ".join(terms)

        # Save to CSV
        output_path = self.output_dir / f"{order}th_order_polynomial_coefficients.csv"

        with open(output_path, 'w') as f:
            f.write(f"# {order}th Order Polynomial Fit\n")
            f.write(f"# {formula}\n")
            f.write("#\n")
            df.to_csv(f, index=False)

        print(f"Polynomial coefficients saved to: {output_path}")
        return output_path

    def plot_ocv_soc_curve(self, coeffs: np.ndarray, order: int = 7) -> Path:
        """
        Plot OCV-SOC curve with original data and polynomial fit.

        Args:
            coeffs: Polynomial coefficients
            order: Polynomial order

        Returns:
            Path to saved plot
        """
        # Generate fine-grained SOC values for smooth curve
        soc_fine = np.linspace(0, 1, 1000)
        ocv_fit = self.evaluate_polynomial(coeffs, soc_fine)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot original data points
        ax.plot(self.SOC_LUT * 100, self.OCV_LUT, 'ro', markersize=8,
                label='Original Data Points', zorder=3)

        # Plot polynomial fit
        ax.plot(soc_fine * 100, ocv_fit, 'b-', linewidth=2,
                label=f'{order}th Order Polynomial Fit', zorder=2)

        # Formatting
        ax.set_xlabel('SOC (%)', fontsize=12)
        ax.set_ylabel('OCV (V)', fontsize=12)
        ax.set_title(f'CRDST Battery OCV-SOC Curve ({order}th Order Polynomial Fit)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xlim(0, 100)

        # Add statistics
        residuals = self.OCV_LUT - self.evaluate_polynomial(coeffs, self.SOC_LUT)
        rmse = np.sqrt(np.mean(residuals**2))
        r_squared = 1 - (np.sum(residuals**2) / np.sum((self.OCV_LUT - np.mean(self.OCV_LUT))**2))

        textstr = f'RMSE: {rmse:.6f} V\n$R^2$: {r_squared:.6f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        # Save plot
        output_path = self.output_dir / f"{order}th_order_polynomial_fit.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"OCV-SOC curve plot saved to: {output_path}")
        print(f"  RMSE: {rmse:.6f} V")
        print(f"  R-squared: {r_squared:.6f}")

        return output_path

    def generate_all(self, polynomial_order: int = 7):
        """
        Generate all OCV-SOC data files.

        Args:
            polynomial_order: Order of polynomial fit (default: 7)
        """
        print("=" * 70)
        print("CRDST OCV-SOC Curve Generation")
        print("=" * 70)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Polynomial order: {polynomial_order}")

        # Fit polynomial
        print("\n1. Fitting polynomial...")
        coeffs = self.fit_polynomial(order=polynomial_order)
        print(f"   Polynomial coefficients (highest to lowest order):")
        for i, coeff in enumerate(coeffs):
            print(f"     p{i} = {coeff:.10e}")

        # Save original data
        print("\n2. Saving original OCV-SOC data...")
        self.save_original_data()

        # Save polynomial coefficients
        print("\n3. Saving polynomial coefficients...")
        self.save_polynomial_coefficients(coeffs, order=polynomial_order)

        # Plot curve
        print("\n4. Generating OCV-SOC curve plot...")
        self.plot_ocv_soc_curve(coeffs, order=polynomial_order)

        print("\n" + "=" * 70)
        print("Generation complete!")
        print("=" * 70)

        # Summary
        print(f"\nGenerated files:")
        print(f"  - Discharge_OCV_SOC_data.csv")
        print(f"  - {polynomial_order}th_order_polynomial_coefficients.csv")
        print(f"  - {polynomial_order}th_order_polynomial_fit.png")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate CRDST OCV-SOC curve data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate OCV-SOC data with default 7th order polynomial
  python create_crdst_ocv_soc.py

  # Specify custom output directory
  python create_crdst_ocv_soc.py -o ./custom_output

  # Use different polynomial order
  python create_crdst_ocv_soc.py --order 5

Output files:
  - Discharge_OCV_SOC_data.csv: Original lookup table
  - 7th_order_polynomial_coefficients.csv: Fitted coefficients
  - 7th_order_polynomial_fit.png: Visualization plot
        """
    )

    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory (default: dataset/processed/CRDST/OCV-SOC/Sample1)')
    parser.add_argument('--order', type=int, default=7,
                        help='Polynomial order (default: 7)')

    args = parser.parse_args()

    # Create generator
    generator = CRDSTOCVGenerator(output_dir=args.output)

    # Generate all files
    generator.generate_all(polynomial_order=args.order)


if __name__ == "__main__":
    main()
