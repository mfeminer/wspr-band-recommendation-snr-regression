"""
WSPR Model Training

Outputs:
- models/best_model_YYYYMMDD_HHMMSS.pkl
- outputs/06_training/run_YYYYMMDD_HHMMSS/training_results.txt
"""

import argparse
import gc
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_DIR / f"06_train_model_{RUN_TIMESTAMP}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='WSPR Model Training')
    parser.add_argument('--input', default='data/spots.20.parquet',
                       help='Input parquet file (default: data/spots.20.parquet)')
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent.resolve()
    os.chdir(project_root)
    
    logger.info("="*70)
    logger.info("WSPR SNR PREDICTION - MODEL TRAINING")
    logger.info("="*70)
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Input: {args.input}")
    logger.info("")
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"ERROR: Input file not found: {input_path}")
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
    
    logger.info("[OK] Input file validated")
    logger.info("")
    
    features_num = ["distance", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    all_bands = [-1, 0, 1, 3, 5, 7, 10, 14, 18, 21, 24, 28, 50]
    all_band_cols = [f"band_{b}" for b in all_bands]
    n_features = len(features_num) + len(all_band_cols)
    
    logger.info(f"Features: {n_features} ({len(features_num)} numeric + {len(all_band_cols)} categorical)")
    logger.info(f"Bands: {all_bands}")
    logger.info("")
    
    total_train = 0
    
    # 5. Train model
    logger.info("="*70)
    logger.info("TRAINING PHASE")
    logger.info("="*70)
    
    sgd_model = SGDRegressor(
        max_iter=10,
        tol=1e-4,
        alpha=0.001,
        learning_rate='adaptive',
        eta0=0.001,
        random_state=42,
        penalty='l2',
        warm_start=True
    )
    
    scaler = StandardScaler()
    
    logger.info("Training SGDRegressor with partial_fit and StandardScaler...")
    logger.info("  Model: SGDRegressor(learning_rate='adaptive', eta0=0.001, alpha=0.001)")
    train_start = datetime.now()
    
    is_scaler_fitted = False
    CHUNK_SIZE = 1_000_000
    
    try:
        if input_path.is_dir():
            part_files = sorted(input_path.glob("part.*.parquet"))
            logger.info(f"  Reading {len(part_files)} partition files...")
        else:
            part_files = [input_path]
            logger.info(f"  Reading single file...")
        
        for file_idx, part_file in enumerate(part_files, 1):
            logger.info(f"  Processing file {file_idx}/{len(part_files)}: {part_file.name}...")
            
            chunk_raw = pd.read_parquet(part_file, columns=["time", "band", "distance", "snr"])
            chunk_rows = len(chunk_raw)
            num_chunks = max(1, (chunk_rows + CHUNK_SIZE - 1) // CHUNK_SIZE)
            
            if num_chunks > 1:
                logger.info(f"    Row group size: {chunk_rows:,} rows → splitting into {num_chunks} mini-batches")
            
            for batch_idx in range(num_chunks):
                start_idx = batch_idx * CHUNK_SIZE
                end_idx = min((batch_idx + 1) * CHUNK_SIZE, chunk_rows)
                chunk = chunk_raw.iloc[start_idx:end_idx][["time", "band", "distance", "snr"]].copy()
                
                if num_chunks > 1:
                    logger.info(f"      Batch {batch_idx+1}/{num_chunks}: rows {start_idx:,} - {end_idx:,} ({len(chunk):,} rows)")
                
                chunk["time"] = pd.to_datetime(chunk["time"], utc=True)
                
                hour = chunk["time"].dt.hour + chunk["time"].dt.minute / 60.0
                hour_rad = 2.0 * np.pi * (hour / 24.0)
                chunk["hour_sin"] = np.sin(hour_rad)
                chunk["hour_cos"] = np.cos(hour_rad)
                
                doy = chunk["time"].dt.dayofyear.astype(float)
                doy_rad = 2.0 * np.pi * (doy / 365.25)
                chunk["doy_sin"] = np.sin(doy_rad)
                chunk["doy_cos"] = np.cos(doy_rad)
                
                X_chunk = chunk[features_num + ["band"]]
                y_chunk = chunk["snr"].astype(float)
                
                X_train_chunk, _, y_train_chunk, _ = train_test_split(
                    X_chunk, y_chunk, test_size=0.25, random_state=42, shuffle=True
                )
                
                total_train += len(X_train_chunk)
                
                X_train_ohe = pd.get_dummies(X_train_chunk, columns=["band"], prefix="band")
                for col in all_band_cols:
                    if col not in X_train_ohe.columns:
                        X_train_ohe[col] = 0
                X_train_ohe = X_train_ohe[features_num + all_band_cols]
                
                if not is_scaler_fitted:
                    X_train_scaled = scaler.fit_transform(X_train_ohe)
                    is_scaler_fitted = True
                else:
                    X_train_scaled = scaler.transform(X_train_ohe)
                
                sgd_model.partial_fit(X_train_scaled, y_train_chunk)
                
                del X_chunk, y_chunk, X_train_chunk, X_train_ohe, X_train_scaled, y_train_chunk, chunk
                gc.collect()
            
            del chunk_raw
            gc.collect()
        
        logger.info(f"[OK] Train set: {total_train:,} rows")
        
        train_duration = (datetime.now() - train_start).total_seconds()
        logger.info(f"[OK] Training completed in {train_duration:.1f}s ({train_duration/60:.1f}min)")
    except Exception as e:
        logger.error(f"[ERROR] Training failed: {e}")
        raise
    
    # 6. Save model and outputs
    logger.info("\nSaving model and results...")
    Path("models").mkdir(exist_ok=True)
    output_dir = Path(f"outputs/06_training/run_{RUN_TIMESTAMP}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = Path(f"models/best_model_{RUN_TIMESTAMP}.pkl")
    scaler_path = Path(f"models/scaler_{RUN_TIMESTAMP}.pkl")
    
    model_data = {
        "model": sgd_model,
        "name": "SGDRegressor",
        "X_columns": features_num + all_band_cols,
        "timestamp": RUN_TIMESTAMP
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    
    logger.info(f"[OK] Model saved to: {model_path}")
    logger.info(f"[OK] Scaler saved to: {scaler_path}")
    
    # Save training results
    results_text = []
    results_text.append("="*70)
    results_text.append("WSPR SNR PREDICTION - MODEL TRAINING RESULTS")
    results_text.append("="*70)
    results_text.append(f"Run ID: {RUN_TIMESTAMP}")
    results_text.append(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results_text.append(f"Training duration: {train_duration:.1f}s ({train_duration/60:.1f}min)")
    results_text.append("")
    results_text.append("MODEL CONFIGURATION:")
    results_text.append("  Algorithm: SGDRegressor (Stochastic Gradient Descent)")
    results_text.append("  Penalty: L2 (Ridge)")
    results_text.append("  Alpha: 0.001")
    results_text.append("  Learning Rate: adaptive")
    results_text.append("  Eta0 (Initial LR): 0.001")
    results_text.append("  Max Iterations per Partition: 10")
    results_text.append("  Feature Scaling: StandardScaler")
    results_text.append("")
    results_text.append("DATASET:")
    results_text.append(f"  Input file: {args.input}")
    results_text.append(f"  Training: {total_train:,} rows (75% of input)")
    results_text.append(f"  Features: {n_features}")
    results_text.append(f"    - Numeric: {len(features_num)} (distance, hour_sin, hour_cos, doy_sin, doy_cos)")
    results_text.append(f"    - Categorical (OHE): {len(all_band_cols)} (band)")
    results_text.append("")
    results_text.append("MEMORY MANAGEMENT:")
    results_text.append("  Method: Row-group based mini-batch training")
    results_text.append("  Batch size: 1M rows")
    results_text.append("")
    results_text.append("SAVED FILES:")
    results_text.append(f"  Model: {model_path}")
    results_text.append(f"  Scaler: {scaler_path}")
    results_text.append("")
    results_text.append("NEXT STEP:")
    results_text.append("  Run 07_evaluate_model.py to evaluate this model")
    results_text.append("")
    results_text.append("="*70)
    
    results_file = output_dir / "training_results.txt"
    with open(results_file, "w") as f:
        f.write("\n".join(results_text))
    logger.info(f"[OK] Training results saved to: {results_file}")
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Run ID: {RUN_TIMESTAMP}")
    logger.info(f"Training data: {total_train:,} rows (75% of input)")
    logger.info(f"Features: {n_features}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Scaler: {scaler_path}")
    logger.info(f"Results: {results_file}")
    logger.info("")
    logger.info("Next step: Run 07_evaluate_model.py to evaluate this model")
    logger.info("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

