"""
WSPR Stratified Sampling
Creates stratified samples from WSPR data while preserving band distribution

Usage:
    python 01_create_samples.py
    python 01_create_samples.py --input spots.full.parquet --percent 20
    python 01_create_samples.py --input data/spots.full.parquet --percent 10

Arguments:
    --input: Input parquet file (default: spots.full.parquet)
    --percent: Sample percentage 1-100 (default: 20)

Output:
    data/spots.{percent}.parquet (e.g., spots.20.parquet for 20%)
"""

import argparse
import gc
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import dask
import dask.dataframe as dd
import pandas as pd

dask.config.set({
    'dataframe.query-planning': False,
    'distributed.worker.memory.target': 15e9,
    'distributed.worker.memory.spill': 17e9,
    'distributed.worker.memory.pause': 18.5e9,
    'distributed.worker.memory.terminate': 20e9
})

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_DIR / f"01_create_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Create stratified sample from WSPR data')
    parser.add_argument('--input', default='data/spots.full.parquet', 
                       help='Input parquet file (default: data/spots.full.parquet)')
    parser.add_argument('--percent', type=float, default=20.0,
                       help='Sample percentage 1-100 (default: 20)')
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("WSPR STRATIFIED SAMPLING")
    logger.info("="*70)
    
    input_path = Path(args.input)
    
    sample_percent = args.percent
    
    if sample_percent < 1 or sample_percent > 100:
        logger.error(f"ERROR: Percent must be between 1-100, got {sample_percent}")
        sys.exit(1)
    
    sample_fraction = sample_percent / 100.0
    
    logger.info(f"Input file: {input_path}")
    logger.info(f"Sample: {sample_percent}%")
    logger.info("")
    
    if not input_path.exists():
        logger.error(f"ERROR: Input file not found: {input_path}")
        logger.error("")
        logger.error("Available files in data/:")
        data_dir = Path("data")
        if data_dir.exists():
            parquet_files = list(data_dir.glob("*.parquet"))
            if parquet_files:
                for f in parquet_files:
                    logger.error(f"  - {f.name}")
            else:
                logger.error("  (no .parquet files found)")
        else:
            logger.error("  (data/ directory not found)")
        sys.exit(1)
    
    try:
        logger.info("Loading dataset...")
        ddf = dd.read_parquet(str(input_path))
    except Exception as e:
        logger.error(f"ERROR: Failed to read parquet file: {e}")
        logger.error(f"File: {input_path}")
        sys.exit(1)
    
    try:
        logger.info("Optimizing data types...")
        ddf["band"] = ddf["band"].astype("int16")
        ddf["distance"] = ddf["distance"].astype("int32")
        ddf["snr"] = ddf["snr"].astype("int8")
        ddf["tx_lat"] = ddf["tx_lat"].astype("float32")
        ddf["tx_lon"] = ddf["tx_lon"].astype("float32")
        ddf["rx_lat"] = ddf["rx_lat"].astype("float32")
        ddf["rx_lon"] = ddf["rx_lon"].astype("float32")
    except Exception as e:
        logger.error(f"ERROR: Failed to optimize data types: {e}")
        logger.error("Dataset might have unexpected columns or types")
        sys.exit(1)
    
    try:
        total_rows = len(ddf)
        logger.info(f"Total rows: {total_rows:,}")
        logger.info(f"Partitions: {ddf.npartitions}")
    except Exception as e:
        logger.error(f"ERROR: Failed to get dataset info: {e}")
        sys.exit(1)
    
    try:
        logger.info("\nComputing band distribution...")
        band_counts = ddf["band"].value_counts().compute().sort_index()
    except Exception as e:
        logger.error(f"ERROR: Failed to compute band distribution: {e}")
        sys.exit(1)
    
    logger.info("\nBAND DISTRIBUTION:")
    for band, count in band_counts.items():
        pct = (count / total_rows * 100)
        logger.info(f"  Band {band:>3d}: {count:>12,} rows ({pct:>5.2f}%)")
    
    logger.info("\n" + "="*70)
    logger.info(f"Creating {sample_percent}% STRATIFIED sample (by band)...")
    logger.info("="*70)
    
    temp_dir = Path("data") / "temp_samples"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Sampling and saving each band separately...")
    logger.info(f"Processing {ddf.npartitions} partitions per band...\n")
    
    try:
        for idx, band_id in enumerate(band_counts.index, 1):
            band_count = band_counts[band_id]
            expected = int(band_count * sample_fraction)
            
            logger.info(f"[{idx:>2d}/{len(band_counts)}] Band {band_id:>3d}: sampling ~{expected:>10,} rows...")
            
            band_chunks = []
            for part_idx in range(ddf.npartitions):
                partition = ddf.get_partition(part_idx)
                
                band_partition = partition[partition["band"] == band_id]
                band_partition_sample = band_partition.sample(frac=sample_fraction, random_state=42)
                
                try:
                    chunk_df = band_partition_sample.compute()
                    if len(chunk_df) > 0:
                        band_chunks.append(chunk_df)
                except Exception as part_error:
                    logger.warning(f"     Partition {part_idx} failed: {part_error}")
                    continue
            
            if band_chunks:
                band_full_sample = pd.concat(band_chunks, ignore_index=True)
                
                band_output = temp_dir / f"band_{band_id:03d}.parquet"
                band_full_sample.to_parquet(str(band_output))
                
                logger.info(f"     Saved {len(band_full_sample):,} rows to temp/band_{band_id:03d}.parquet")
                del band_full_sample
            else:
                logger.warning(f"     No data for band {band_id}")
            
            del band_chunks
            gc.collect()
                
    except Exception as e:
        logger.error(f"ERROR: Failed during stratified sampling: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)
    
    output_filename = f"spots.{int(sample_percent)}.parquet"
    output_path = Path("data") / output_filename
    
    try:
        logger.info(f"\nMerging all band samples into {output_path}...")
        all_band_samples = dd.read_parquet(str(temp_dir / "band_*.parquet"))
        all_band_samples.to_parquet(str(output_path), overwrite=True)
        logger.info(f"[OK] Saved: {output_path}")
        logger.info(f"[OK] Sample rate: {sample_percent}%")
    except Exception as e:
        logger.error(f"ERROR: Failed to merge samples: {e}")
        logger.error(f"Output path: {output_path}")
        sys.exit(1)
    finally:
        logger.info("\nCleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("[OK] Temporary files removed")
    
    logger.info("\n" + "="*70)
    logger.info("VERIFICATION: Band distribution preserved?")
    logger.info("="*70)
    
    sample_total = int(total_rows * sample_fraction)
    
    try:
        logger.info(f"\nLoading {output_filename} for verification...")
        sample_check = dd.read_parquet(str(output_path))
        sample_band_counts_series = sample_check["band"].value_counts()
        sample_band_counts, sample_total = dask.compute(sample_band_counts_series, sample_check.shape[0])
        sample_band_counts = sample_band_counts.sort_index()
        
        logger.info(f"Sample size: {sample_total:,} rows")
        logger.info(f"\nCOMPARISON (Full vs {sample_percent}% Sample):")
        logger.info(f"{'Band':>6s} {'Full %':>8s} {'Sample %':>10s} {'Diff':>8s}")
        logger.info("-" * 40)
        
        max_diff = 0.0
        for band_id in band_counts.index:
            full_pct = (band_counts[band_id] / total_rows * 100)
            sample_pct = (sample_band_counts[band_id] / sample_total * 100)
            diff = abs(full_pct - sample_pct)
            max_diff = max(max_diff, diff)
            logger.info(f"{band_id:>6d} {full_pct:>7.2f}% {sample_pct:>9.2f}% {diff:>7.3f}%")
        
        if max_diff < 0.1:
            logger.info(f"\n[OK] Stratification excellent (max diff: {max_diff:.3f}%)")
        elif max_diff < 0.5:
            logger.info(f"\n[OK] Stratification good (max diff: {max_diff:.3f}%)")
        else:
            logger.warning(f"\n[WARNING] Stratification acceptable (max diff: {max_diff:.3f}%)")
    except Exception as e:
        logger.error(f"ERROR: Failed to verify sample: {e}")
        logger.error("Sample was created but verification failed")
    
    logger.info("\n" + "="*70)
    logger.info("SAMPLING COMPLETE")
    logger.info("="*70)
    logger.info(f"Input:  {input_path.name}")
    logger.info(f"Output: {output_path.name}")
    logger.info(f"Rows:   {total_rows:,} → ~{sample_total:,} ({sample_percent}%)")
    logger.info(f"Method: Stratified by band (random_state=42)")
    logger.info("="*70)

if __name__ == "__main__":
    main()

