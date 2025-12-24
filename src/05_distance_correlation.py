"""
WSPR Distance vs SNR Correlation

Outputs:
- outputs/05_distance/correlation_matrix.png
- outputs/05_distance/distance_vs_snr_scatter.png
- outputs/05_distance/distance_vs_snr_hexbin.png
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import dask
import dask.dataframe as dd
import matplotlib.pyplot as plt
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

OUTPUT_DIR = Path("outputs/05_distance")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_DIR / f"05_distance_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
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
    ddf = ddf.dropna(subset=["snr", "distance", "band"])
    except Exception as e:
        logger.error(f"ERROR: Failed to process data: {e}")
        sys.exit(1)
    
    logger.info(f"Dataset size: ~{len(ddf):,} rows")
    return ddf

def plot_correlation_matrix(ddf):
    logger.info("Computing correlation matrix...")
    
    try:
        chunks = []
        for i in range(ddf.npartitions):
            logger.info(f"  Processing partition {i+1}/{ddf.npartitions}...")
            partition = ddf.get_partition(i)[["distance", "snr", "band"]].compute()
            chunks.append(partition)
            del partition
        
        logger.info("Combining partitions...")
        df_sample = pd.concat(chunks, ignore_index=True)
        logger.info(f"Computing correlation on {len(df_sample):,} rows...")
    corr_matrix = df_sample.corr()
    
        del chunks, df_sample
    except Exception as e:
        logger.error(f"ERROR: Failed to compute correlation matrix: {e}")
        import sys
        sys.exit(1)
    
    try:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {OUTPUT_DIR / 'correlation_matrix.png'}")
    plt.close()
    except Exception as e:
        logger.error(f"ERROR: Failed to create correlation matrix plot: {e}")
        import sys
        sys.exit(1)
    
    print("\nCORRELATION MATRIX:")
    print(corr_matrix)

def plot_distance_vs_snr_scatter(ddf):
    logger.info("Creating distance vs SNR scatter plot...")
    
    try:
        chunks = []
        for i in range(ddf.npartitions):
            logger.info(f"  Processing partition {i+1}/{ddf.npartitions}...")
            partition = ddf.get_partition(i)[["distance", "snr"]].compute()
            chunks.append(partition)
            del partition
        
        logger.info("Combining partitions...")
        df_scatter = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(df_scatter):,} rows for scatter plot")
    
        del chunks
    except Exception as e:
        logger.error(f"ERROR: Failed to load data for scatter plot: {e}")
        import sys
        sys.exit(1)
    
    try:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.scatter(df_scatter['distance'], df_scatter['snr'], 
               alpha=0.3, s=1, c='steelblue', rasterized=True)
    
    ax.set_xlabel("Distance (km)", fontsize=12, fontweight='bold')
    ax.set_ylabel("SNR (dB)", fontsize=12, fontweight='bold')
        ax.set_title(f"Distance vs SNR ({len(df_scatter):,} points)", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distance_vs_snr_scatter.png", dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {OUTPUT_DIR / 'distance_vs_snr_scatter.png'}")
    plt.close()
    except Exception as e:
        logger.error(f"ERROR: Failed to create scatter plot: {e}")
        import sys
        sys.exit(1)

def plot_distance_vs_snr_hexbin(ddf):
    logger.info("Creating distance vs SNR hexbin plot...")
    
    try:
        chunks = []
        for i in range(ddf.npartitions):
            logger.info(f"  Processing partition {i+1}/{ddf.npartitions}...")
            partition = ddf.get_partition(i)[["distance", "snr"]].compute()
            chunks.append(partition)
            del partition
        
        logger.info("Combining partitions...")
        df_hexbin = pd.concat(chunks, ignore_index=True)
        logger.info(f"Using {len(df_hexbin):,} rows for hexbin plot")
    
        del chunks
    except Exception as e:
        logger.error(f"ERROR: Failed to load data for hexbin plot: {e}")
        import sys
        sys.exit(1)
    
    try:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    hexbin = ax.hexbin(df_hexbin['distance'], df_hexbin['snr'], 
                       gridsize=50, cmap='YlOrRd', mincnt=1, bins='log')
    
    ax.set_xlabel("Distance (km)", fontsize=12, fontweight='bold')
    ax.set_ylabel("SNR (dB)", fontsize=12, fontweight='bold')
        ax.set_title(f"Distance vs SNR Density ({len(df_hexbin):,} points)", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.axhline(0, color='blue', linestyle='--', linewidth=0.8, alpha=0.5)
    
    cbar = plt.colorbar(hexbin, ax=ax)
    cbar.set_label('Count (log scale)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distance_vs_snr_hexbin.png", dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {OUTPUT_DIR / 'distance_vs_snr_hexbin.png'}")
    plt.close()
    except Exception as e:
        logger.error(f"ERROR: Failed to create hexbin plot: {e}")
        import sys
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='WSPR Distance vs SNR Correlation')
    parser.add_argument('--input', default='data/spots.20.parquet',
                       help='Input parquet file (default: data/spots.20.parquet)')
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("WSPR DISTANCE vs SNR CORRELATION ANALYSIS")
    logger.info("="*70)
    logger.info(f"Input: {args.input}")
    logger.info("")
    
    ddf = load_data(args.input)
    plot_correlation_matrix(ddf)
    plot_distance_vs_snr_scatter(ddf)
    plot_distance_vs_snr_hexbin(ddf)
    
    logger.info("="*70)
    logger.info("CORRELATION ANALYSIS COMPLETE")
    logger.info(f"Outputs saved to: {OUTPUT_DIR}")
    logger.info("="*70)

if __name__ == "__main__":
    main()
