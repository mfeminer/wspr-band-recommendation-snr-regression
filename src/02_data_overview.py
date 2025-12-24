"""
WSPR Data Overview

Outputs:
- outputs/02_overview/statistics.txt
- outputs/02_overview/band_distribution.png
- outputs/02_overview/band_counts.csv
"""

import argparse
import csv
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

OUTPUT_DIR = Path("outputs/02_overview")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_DIR / f"02_data_overview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BAND_LABELS = {
    -1: "LF", 0: "MF", 1: "160m", 3: "80m", 5: "60m",
    7: "40m", 10: "30m", 14: "20m", 18: "17m", 21: "15m",
    24: "12m", 28: "10m", 50: "6m"
}

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
    
    logger.info(f"Loaded {ddf.npartitions} partitions")
    logger.info(f"Dataset size: ~{len(ddf):,} rows")
    
    return ddf

def compute_statistics(ddf):
    logger.info("Computing basic statistics...")
    
    output = []
    output.append("="*70)
    output.append("WSPR DATA OVERVIEW")
    output.append("="*70)
    output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("")
    
    total_rows = len(ddf)
    output.append(f"Total Rows: {total_rows:,}")
    output.append(f"Total Columns: {len(ddf.columns)}")
    output.append("")
    
    output.append("COLUMN TYPES:")
    for col, dtype in ddf.dtypes.items():
        output.append(f"  {col:12s}: {dtype}")
    output.append("")
    
    logger.info("Computing statistics...")
    
    import dask
    snr_stats = dask.compute(
        ddf["snr"].mean(),
        ddf["snr"].std(),
        ddf["snr"].min(),
        ddf["snr"].max()
    )
    snr_mean, snr_std, snr_min, snr_max = snr_stats
    
    output.append("SNR STATISTICS:")
    output.append(f"  Mean:    {snr_mean:.2f} dB")
    output.append(f"  Std Dev: {snr_std:.2f} dB")
    output.append(f"  Min:     {snr_min} dB")
    output.append(f"  Max:     {snr_max} dB")
    output.append("")
    
    dist_stats = dask.compute(
        ddf["distance"].mean(),
        ddf["distance"].std(),
        ddf["distance"].min(),
        ddf["distance"].max()
    )
    dist_mean, dist_std, dist_min, dist_max = dist_stats
    
    output.append("DISTANCE STATISTICS:")
    output.append(f"  Mean:    {dist_mean:.0f} km")
    output.append(f"  Std Dev: {dist_std:.0f} km")
    output.append(f"  Min:     {dist_min} km")
    output.append(f"  Max:     {dist_max} km")
    output.append("")
    
    logger.info("Checking missing values...")
    missing = ddf.isnull().sum().compute()
    output.append("MISSING VALUES:")
    if missing.sum() == 0:
        output.append("  No missing values found")
    else:
        for col, count in missing.items():
            if count > 0:
                pct = (count / total_rows * 100)
                output.append(f"  {col}: {count:,} ({pct:.4f}%)")
    output.append("")
    output.append("="*70)
    
    stats_text = "\n".join(output)
    with open(OUTPUT_DIR / "statistics.txt", "w") as f:
        f.write(stats_text)
    
    logger.info(f"Saved: {OUTPUT_DIR / 'statistics.txt'}")
    print("\n" + stats_text)
    
    return total_rows

def analyze_bands(ddf):
    logger.info("Computing band distribution...")
    band_counts = ddf["band"].value_counts().compute().sort_index()
    
    band_data = []
    total = len(ddf)
    
    for band, count in band_counts.items():
        label = BAND_LABELS.get(band, str(band))
        pct = (count / total * 100)
        band_data.append({
            'Band': band,
            'Label': label,
            'Count': count,
            'Percentage': pct
        })
    
    with open(OUTPUT_DIR / "band_counts.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Band', 'Label', 'Count', 'Percentage'])
        writer.writeheader()
        writer.writerows(band_data)
    
    logger.info(f"Saved: {OUTPUT_DIR / 'band_counts.csv'}")
    
    print("\nBAND DISTRIBUTION:")
    for row in band_data:
        print(f"  {row['Label']:6s}: {row['Count']:>12,} spots ({row['Percentage']:>5.1f}%)")
    
    return band_data

def plot_band_distribution(band_data):
    logger.info("Creating band distribution plot...")
    
    labels = [row['Label'] for row in band_data]
    counts = [row['Count'] for row in band_data]
    percentages = [row['Percentage'] for row in band_data]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars = ax.bar(labels, counts, 
                   color='steelblue', edgecolor='black', linewidth=0.7)
    
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel("Band", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Spots", fontsize=12, fontweight='bold')
    ax.set_title("WSPR Spot Distribution by Band (2+ Billion Spots)", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1e6)}M' if x >= 1e6 else f'{int(x/1e3)}K'))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "band_distribution.png", dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {OUTPUT_DIR / 'band_distribution.png'}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='WSPR Data Overview Analysis')
    parser.add_argument('--input', default='data/spots.20.parquet',
                       help='Input parquet file (default: data/spots.20.parquet)')
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("WSPR DATA OVERVIEW ANALYSIS")
    logger.info("="*70)
    logger.info(f"Input: {args.input}")
    logger.info("")
    
    ddf = load_data(args.input)
    total_rows = compute_statistics(ddf)
    band_data = analyze_bands(ddf)
    plot_band_distribution(band_data)
    
    logger.info("="*70)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Total rows analyzed: {total_rows:,}")
    logger.info(f"Outputs saved to: {OUTPUT_DIR}")
    logger.info("="*70)

if __name__ == "__main__":
    main()
