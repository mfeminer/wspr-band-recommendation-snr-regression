"""
WSPR Data Quality Checks

Outputs:
- outputs/03_quality/quality_report.txt
- outputs/03_quality/outlier_analysis.png
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import dask
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(Path(__file__).parent.parent)

dask.config.set({
    'dataframe.query-planning': False,
    'distributed.worker.memory.target': 15e9,
    'distributed.worker.memory.spill': 17e9,
    'distributed.worker.memory.pause': 18.5e9,
    'distributed.worker.memory.terminate': 20e9
})

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

OUTPUT_DIR = Path("outputs/03_quality")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_DIR / f"03_data_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(data_path):
    import sys
    
    data_path = Path(data_path)
    if not data_path.exists():
        logger.error(f"ERROR: Input file not found: {data_path}")
        logger.error("")
        logger.error("Available files in data/:")
        data_dir = Path("data")
        if data_dir.exists():
            parquet_files = list(data_dir.glob("*.parquet"))
            if parquet_files:
                for f in sorted(parquet_files):
                    logger.error(f"  - {f.name}")
            else:
                logger.error("  (no .parquet files found)")
        sys.exit(1)
    
    try:
        logger.info(f"Loading data from {data_path}...")
        ddf = dd.read_parquet(str(data_path))
    except Exception as e:
        logger.error(f"ERROR: Failed to read parquet file: {e}")
        sys.exit(1)
    
    try:
        ddf["band"] = ddf["band"].astype("int16")
        ddf["distance"] = ddf["distance"].astype("int32")
        ddf["snr"] = ddf["snr"].astype("int8")
        ddf["tx_lat"] = ddf["tx_lat"].astype("float32")
        ddf["tx_lon"] = ddf["tx_lon"].astype("float32")
        ddf["rx_lat"] = ddf["rx_lat"].astype("float32")
        ddf["rx_lon"] = ddf["rx_lon"].astype("float32")
    except Exception as e:
        logger.error(f"ERROR: Failed to optimize data types: {e}")
        sys.exit(1)
    
    logger.info(f"Dataset size: ~{len(ddf):,} rows")
    return ddf

def check_data_quality(ddf):
    logger.info("Running data quality checks...")
    
    output = []
    output.append("="*70)
    output.append("WSPR DATA QUALITY REPORT")
    output.append("="*70)
    output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("")
    
    total_rows = len(ddf)
    output.append(f"Total Rows: {total_rows:,}")
    output.append("")
    
    logger.info("1. SNR Range Check...")
    snr_min = ddf["snr"].min().compute()
    snr_max = ddf["snr"].max().compute()
    invalid_snr = ((ddf["snr"] < -50) | (ddf["snr"] > 50)).sum().compute()
    
    output.append("1. SNR RANGE CHECK")
    output.append(f"   Range: {snr_min} to {snr_max} dB")
    output.append(f"   Invalid SNR (outside -50 to +50 dB): {invalid_snr:,} ({invalid_snr/total_rows*100:.4f}%)")
    output.append("")
    
    logger.info("2. Distance Sanity Check...")
    dist_min = ddf["distance"].min().compute()
    dist_max = ddf["distance"].max().compute()
    invalid_dist = ((ddf["distance"] < 0) | (ddf["distance"] > 25000)).sum().compute()
    
    output.append("2. DISTANCE SANITY CHECK")
    output.append(f"   Range: {dist_min} to {dist_max} km")
    output.append(f"   Invalid distance (< 0 or > 25000 km): {invalid_dist:,}")
    output.append("")
    
    logger.info("3. Outlier Detection (IQR Method)...")
    Q1 = ddf["snr"].quantile(0.25).compute()
    Q3 = ddf["snr"].quantile(0.75).compute()
    IQR = Q3 - Q1
    outlier_low = Q1 - 1.5 * IQR
    outlier_high = Q3 + 1.5 * IQR
    snr_outliers = ((ddf["snr"] < outlier_low) | (ddf["snr"] > outlier_high)).sum().compute()
    
    output.append("3. OUTLIER DETECTION (IQR Method for SNR)")
    output.append(f"   Q1: {Q1:.1f} dB, Q3: {Q3:.1f} dB, IQR: {IQR:.1f} dB")
    output.append(f"   IQR bounds: [{outlier_low:.1f}, {outlier_high:.1f}] dB")
    output.append(f"   SNR outliers: {snr_outliers:,} ({snr_outliers/total_rows*100:.2f}%)")
    output.append("   Note: Outliers are KEPT - they represent real weak/strong signals")
    output.append("")
    
    logger.info("4. Coordinate Validity Check...")
    invalid_coords = (
        (ddf["tx_lat"].abs() > 90) | 
        (ddf["tx_lon"].abs() > 180) | 
        (ddf["rx_lat"].abs() > 90) | 
        (ddf["rx_lon"].abs() > 180)
    ).sum().compute()
    
    output.append("4. COORDINATE VALIDITY")
    output.append(f"   Invalid coordinates: {invalid_coords:,}")
    output.append("")
    
    logger.info("5. Duplicate Check (Sampling 1M rows)...")
    sample_for_dup = ddf.head(1000000, npartitions=-1)
    duplicates_in_sample = sample_for_dup.duplicated().sum()
    
    output.append("5. DUPLICATE CHECK (1M Sample)")
    output.append(f"   Duplicate rows in sample: {duplicates_in_sample:,}")
    output.append("   Note: Full duplicate check skipped for performance (2B+ rows)")
    output.append("")
    
    total_issues = invalid_snr + invalid_dist + invalid_coords
    output.append("="*70)
    output.append("SUMMARY")
    output.append("="*70)
    output.append(f"Total Issues Found: {total_issues:,} rows ({total_issues/total_rows*100:.4f}%)")
    output.append(f"Data Quality: {'EXCELLENT' if total_issues < 100000 else 'GOOD' if total_issues < 1000000 else 'ACCEPTABLE'}")
    output.append("="*70)
    
    report_text = "\n".join(output)
    with open(OUTPUT_DIR / "quality_report.txt", "w") as f:
        f.write(report_text)
    
    logger.info(f"Saved: {OUTPUT_DIR / 'quality_report.txt'}")
    print("\n" + report_text)
    
    return {
        'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
        'outlier_low': outlier_low, 'outlier_high': outlier_high,
        'snr_min': snr_min, 'snr_max': snr_max
    }

def plot_outlier_analysis(ddf, stats):
    logger.info("Creating outlier visualization...")
    
    sample = ddf["snr"].sample(frac=0.01, random_state=42).compute()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(sample, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(stats['Q1'], color='orange', linestyle='--', linewidth=2, label=f'Q1 ({stats["Q1"]:.1f} dB)')
    axes[0].axvline(stats['Q3'], color='orange', linestyle='--', linewidth=2, label=f'Q3 ({stats["Q3"]:.1f} dB)')
    axes[0].axvline(stats['outlier_low'], color='red', linestyle='--', linewidth=2, label=f'Lower ({stats["outlier_low"]:.1f} dB)')
    axes[0].axvline(stats['outlier_high'], color='red', linestyle='--', linewidth=2, label=f'Upper ({stats["outlier_high"]:.1f} dB)')
    axes[0].set_xlabel("SNR (dB)", fontweight='bold')
    axes[0].set_ylabel("Frequency (1% Sample)", fontweight='bold')
    axes[0].set_title("SNR Distribution with IQR Bounds", fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    boxplot_data = sample.values
    bp = axes[1].boxplot([boxplot_data], vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    axes[1].set_ylabel("SNR (dB)", fontweight='bold')
    axes[1].set_title("SNR Box Plot (Shows Outliers)", fontweight='bold')
    axes[1].set_xticklabels(['SNR'])
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "outlier_analysis.png", dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {OUTPUT_DIR / 'outlier_analysis.png'}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='WSPR Data Quality Analysis')
    parser.add_argument('--input', default='data/spots.20.parquet',
                       help='Input parquet file (default: data/spots.20.parquet)')
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("WSPR DATA QUALITY ANALYSIS")
    logger.info("="*70)
    logger.info(f"Input: {args.input}")
    logger.info("")
    
    ddf = load_data(args.input)
    stats = check_data_quality(ddf)
    plot_outlier_analysis(ddf, stats)
    
    logger.info("="*70)
    logger.info("QUALITY ANALYSIS COMPLETE")
    logger.info(f"Outputs saved to: {OUTPUT_DIR}")
    logger.info("="*70)

if __name__ == "__main__":
    main()
