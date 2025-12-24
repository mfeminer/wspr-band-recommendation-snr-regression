"""
WSPR Band-wise SNR Analysis

Outputs:
- outputs/04_band_snr/band_statistics.csv
- outputs/04_band_snr/band_snr_boxplot.png
- outputs/04_band_snr/snr_histogram_by_band.png
"""

import argparse
import gc
import logging
import os
from datetime import datetime
from pathlib import Path

import dask
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

OUTPUT_DIR = Path("outputs/04_band_snr")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_DIR / f"04_band_snr_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BAND_LABELS = {
    -1: "LF", 0: "MF", 1: "160m", 3: "80m", 5: "60m",
    7: "40m", 10: "30m", 14: "20m", 18: "17m", 21: "15m",
    24: "12m", 28: "10m", 50: "6m"
}

MAJOR_BANDS = [7, 10, 14, 18, 21, 28]

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
        ddf["snr"] = ddf["snr"].astype("int8")
        ddf = ddf.dropna(subset=["snr", "band"])
    except Exception as e:
        logger.error(f"ERROR: Failed to process data: {e}")
        sys.exit(1)
    
    logger.info(f"Dataset size: ~{len(ddf):,} rows")
    return ddf

def compute_band_statistics(ddf):
    logger.info("Computing band-wise SNR statistics...")
    
    try:
        all_band_stats = []
        
        for i in range(ddf.npartitions):
            logger.info(f"  Processing partition {i+1}/{ddf.npartitions}...")
            partition = ddf.get_partition(i)[["band", "snr"]].compute()
            partition["snr"] = partition["snr"].astype(float)
            
            part_stats = partition.groupby("band").agg(
                count=('snr', 'count'),
                sum_val=('snr', 'sum'),
                sum_sq=('snr', lambda x: (x**2).sum()),
                min_val=('snr', 'min'),
                max_val=('snr', 'max')
            ).reset_index()
            
            all_band_stats.append(part_stats)
            del partition
            gc.collect()
        
        combined = pd.concat(all_band_stats, ignore_index=True)
        
        final_stats = combined.groupby("band").apply(lambda g: pd.Series({
            'count': g['count'].sum(),
            'mean': g['sum_val'].sum() / g['count'].sum(),
            'std': np.sqrt(max(0, (g['sum_sq'].sum() / g['count'].sum()) - (g['sum_val'].sum() / g['count'].sum())**2)),
            'min': g['min_val'].min(),
            'max': g['max_val'].max()
        }), include_groups=False).reset_index()
        
        final_stats['label'] = final_stats['band'].map(BAND_LABELS)
        final_stats = final_stats.sort_values('count', ascending=False)
        
        band_stats = final_stats
    except Exception as e:
        logger.error(f"ERROR: Failed to compute band statistics: {e}")
        import sys
        sys.exit(1)
    
    try:
        band_stats.to_csv(OUTPUT_DIR / "band_statistics.csv", index=False)
        logger.info(f"Saved: {OUTPUT_DIR / 'band_statistics.csv'}")
    except Exception as e:
        logger.error(f"ERROR: Failed to save band statistics: {e}")
        import sys
        sys.exit(1)
    
    print("\nBAND-WISE SNR STATISTICS:")
    print(f"{'Band':>8s} {'Count':>12s} {'Mean':>8s} {'Std':>8s} {'Min':>6s} {'Max':>6s}")
    print("-" * 60)
    for _, row in band_stats.iterrows():
        print(f"{row['label']:>8s} {int(row['count']):>12,} {row['mean']:>8.2f} {row['std']:>8.2f} {int(row['min']):>6d} {int(row['max']):>6d}")
    
    return band_stats

def plot_band_boxplot(df_sample):
    logger.info("Creating SNR box plot by band...")
    
    try:
        df_major = df_sample[df_sample['band'].isin(MAJOR_BANDS)].copy()
        df_major['band_label'] = df_major['band'].map(BAND_LABELS)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        band_order = [BAND_LABELS[b] for b in sorted(MAJOR_BANDS)]
        sns.boxplot(data=df_major, x="band_label", y="snr", hue="band_label",
                    order=band_order, palette="Set2", ax=ax, legend=False)
        
        ax.set_xlabel("Band", fontsize=12, fontweight='bold')
        ax.set_ylabel("SNR (dB)", fontsize=12, fontweight='bold')
        ax.set_title("SNR Distribution by Band (Box Plot)", 
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "band_snr_boxplot.png", dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {OUTPUT_DIR / 'band_snr_boxplot.png'}")
        plt.close()
        
        del df_major
        gc.collect()
    except Exception as e:
        logger.error(f"ERROR: Failed to create box plot: {e}")
        import sys
        sys.exit(1)

def plot_snr_histogram_by_band(df_sample):
    logger.info("Creating SNR histogram by band...")
    
    try:
        df_major = df_sample[df_sample['band'].isin(MAJOR_BANDS)][['band', 'snr']].copy()
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, band in enumerate(sorted(MAJOR_BANDS)):
            logger.info(f"  Creating histogram for {BAND_LABELS[band]}...")
            label = BAND_LABELS[band]
            band_data = df_major[df_major['band'] == band]
            data = band_data['snr'].values
            
            axes[idx].hist(data, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            axes[idx].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.1f} dB')
            axes[idx].set_xlabel("SNR (dB)", fontweight='bold')
            axes[idx].set_ylabel("Frequency", fontweight='bold')
            axes[idx].set_title(f"{label} ({len(data):,} spots)", fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(axis='y', alpha=0.3)
            
            del band_data, data
        
        plt.suptitle("SNR Histograms by Band", fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "snr_histogram_by_band.png", dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {OUTPUT_DIR / 'snr_histogram_by_band.png'}")
        plt.close()
        
        del df_major
        gc.collect()
    except Exception as e:
        logger.error(f"ERROR: Failed to create histograms: {e}")
        import sys
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='WSPR Band-wise SNR Analysis')
    parser.add_argument('--input', default='data/spots.20.parquet',
                       help='Input parquet file (default: data/spots.20.parquet)')
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("WSPR BAND-WISE SNR ANALYSIS")
    logger.info("="*70)
    logger.info(f"Input: {args.input}")
    logger.info("")
    
    ddf = load_data(args.input)
    band_stats = compute_band_statistics(ddf)
    
    try:
        logger.info("Loading data for visualizations...")
        chunks = []
        for i in range(ddf.npartitions):
            logger.info(f"  Processing partition {i+1}/{ddf.npartitions}...")
            partition = ddf.get_partition(i)[["band", "snr"]].compute()
            chunks.append(partition)
            del partition
            gc.collect()
        
        logger.info("Combining partitions...")
        df_sample = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(df_sample):,} rows for plots")
        
        del chunks
        gc.collect()
    except Exception as e:
        logger.error(f"ERROR: Failed to load data for visualizations: {e}")
        import sys
        sys.exit(1)
    
    plot_band_boxplot(df_sample)
    plot_snr_histogram_by_band(df_sample)
    
    logger.info("="*70)
    logger.info("BAND SNR ANALYSIS COMPLETE")
    logger.info(f"Analyzed {len(band_stats)} bands")
    logger.info(f"Outputs saved to: {OUTPUT_DIR}")
    logger.info("="*70)

if __name__ == "__main__":
    main()
