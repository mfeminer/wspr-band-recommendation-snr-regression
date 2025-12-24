"""
WSPR Model Evaluation

Outputs:
- outputs/07_evaluation/run_YYYYMMDD_HHMMSS/evaluation_results.txt
- outputs/07_evaluation/run_YYYYMMDD_HHMMSS/predictions_vs_actual.png
- outputs/07_evaluation/run_YYYYMMDD_HHMMSS/residuals_plot.png
- outputs/07_evaluation/run_YYYYMMDD_HHMMSS/band_wise_performance.png
"""

import argparse
import gc
import glob
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

os.chdir(Path(__file__).parent.parent)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_DIR / f"07_evaluate_model_{RUN_TIMESTAMP}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BAND_LABELS = {
    -1: "LF", 0: "MF", 1: "160m", 3: "80m", 5: "60m",
    7: "40m", 10: "30m", 14: "20m", 18: "17m", 21: "15m",
    24: "12m", 28: "10m", 50: "6m"
}

def find_latest_model():
    model_files = sorted(glob.glob("models/best_model_*.pkl"), reverse=True)
    if model_files:
        return model_files[0]
    return None

def load_model(model_path=None):
    logger.info("Loading trained model...")
    
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            logger.error("ERROR: No trained model found in models/")
            logger.error("Please run 06_train_model.py first to train the model.")
            sys.exit(1)
        logger.info(f"Using latest model: {model_path}")
    else:
        model_path = Path(model_path)
        if not model_path.exists():
            logger.error(f"ERROR: Model file not found: {model_path}")
            sys.exit(1)
    
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        logger.info(f"[OK] Model loaded: {model_path}")
        return model_data, model_path
    except Exception as e:
        logger.error(f"ERROR: Failed to load model: {e}")
        sys.exit(1)

def load_scaler(scaler_path=None, model_timestamp=None):
    logger.info("Loading scaler...")
    
    if scaler_path is None:
        if model_timestamp:
            scaler_path = f"models/scaler_{model_timestamp}.pkl"
        else:
            scaler_files = sorted(glob.glob("models/scaler_*.pkl"), reverse=True)
            if scaler_files:
                scaler_path = scaler_files[0]
            else:
                logger.error("ERROR: No scaler found in models/")
                logger.error("Please run 06_train_model.py first to train the model.")
                sys.exit(1)
        logger.info(f"Using scaler: {scaler_path}")
    else:
        scaler_path = Path(scaler_path)
        if not scaler_path.exists():
            logger.error(f"ERROR: Scaler file not found: {scaler_path}")
            sys.exit(1)
    
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info(f"[OK] Scaler loaded: {scaler_path}")
        return scaler
    except Exception as e:
        logger.error(f"ERROR: Failed to load scaler: {e}")
        sys.exit(1)

def check_data_file(data_path):
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
    
    return data_path

def evaluate_model(model, scaler, input_path, X_columns):
    try:
        logger.info("Computing predictions...")
        
        if input_path.is_dir():
            part_files = sorted(input_path.glob("part.*.parquet"))
            logger.info(f"  Reading {len(part_files)} partition files...")
        else:
            part_files = [input_path]
            logger.info(f"  Reading single file...")
        
        CHUNK_SIZE = 1_000_000
        
        total_count = 0
        sum_abs_error = 0.0
        sum_y = 0.0
        sum_y_squared = 0.0
        sum_pred = 0.0
        sum_squared_error = 0.0
        
        sample_y = []
        sample_pred = []
        sample_bands = []
        SAMPLE_SIZE = 100000
        sample_interval = 1
        
        for file_idx, part_file in enumerate(part_files, 1):
            logger.info(f"  Processing file {file_idx}/{len(part_files)}: {part_file.name}...")
            
            chunk_raw = pd.read_parquet(part_file, columns=["time", "band", "distance", "snr"])
            chunk_rows = len(chunk_raw)
            num_chunks = max(1, (chunk_rows + CHUNK_SIZE - 1) // CHUNK_SIZE)
            
            if num_chunks > 1:
                logger.info(f"    File size: {chunk_rows:,} rows → splitting into {num_chunks} mini-batches")
            
            for batch_idx in range(num_chunks):
                start_idx = batch_idx * CHUNK_SIZE
                end_idx = min((batch_idx + 1) * CHUNK_SIZE, chunk_rows)
                df_part = chunk_raw.iloc[start_idx:end_idx][["time", "band", "distance", "snr"]].copy()
                
                df_part["time"] = pd.to_datetime(df_part["time"], utc=True)
                
                hour = df_part["time"].dt.hour + df_part["time"].dt.minute / 60.0
                hour_rad = 2.0 * np.pi * (hour / 24.0)
                df_part["hour_sin"] = np.sin(hour_rad)
                df_part["hour_cos"] = np.cos(hour_rad)
                
                doy = df_part["time"].dt.dayofyear.astype(float)
                doy_rad = 2.0 * np.pi * (doy / 365.25)
                df_part["doy_sin"] = np.sin(doy_rad)
                df_part["doy_cos"] = np.cos(doy_rad)
                
                bands_part = df_part["band"].map(BAND_LABELS).values
                y_part = df_part["snr"].values.astype(float)
                
                df_ohe_part = pd.get_dummies(df_part, columns=["band"], prefix="band")
                
                for col in X_columns:
                    if col not in df_ohe_part.columns:
                        df_ohe_part[col] = 0
                
                X_part = df_ohe_part[X_columns]
                X_part_scaled = scaler.transform(X_part)
                
                y_pred_part = model.predict(X_part_scaled)
                
                total_count += len(y_part)
                sum_abs_error += np.sum(np.abs(y_part - y_pred_part))
                sum_y += np.sum(y_part)
                sum_y_squared += np.sum(y_part ** 2)
                sum_pred += np.sum(y_pred_part)
                sum_squared_error += np.sum((y_part - y_pred_part) ** 2)
                
                if len(sample_y) < SAMPLE_SIZE:
                    sample_indices = np.arange(0, len(y_part), sample_interval)
                    sample_y.extend(y_part[sample_indices])
                    sample_pred.extend(y_pred_part[sample_indices])
                    sample_bands.extend(bands_part[sample_indices])
                
                del df_part, df_ohe_part, X_part, X_part_scaled, y_part, y_pred_part, bands_part
                gc.collect()
            
            del chunk_raw
            gc.collect()
        
        logger.info("Computing metrics...")
        mae = sum_abs_error / total_count
        
        mean_y = sum_y / total_count
        ss_tot = sum_y_squared - (sum_y ** 2) / total_count
        ss_res = sum_squared_error
        r2 = 1 - (ss_res / ss_tot)
        
        y = np.array(sample_y[:SAMPLE_SIZE])
        y_pred = np.array(sample_pred[:SAMPLE_SIZE])
        bands = np.array(sample_bands[:SAMPLE_SIZE])
        residuals = y - y_pred
        
        return y, y_pred, bands, mae, r2, residuals, total_count
    except Exception as e:
        logger.error(f"ERROR: Failed to evaluate model: {e}")
        sys.exit(1)

def save_results(mae, r2, sample_size, output_dir, model_info, input_file):
    try:
        output = []
        output.append("="*70)
        output.append("WSPR MODEL EVALUATION RESULTS")
        output.append("="*70)
        output.append(f"Evaluation ID: {RUN_TIMESTAMP}")
        output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("")
        output.append("MODEL INFORMATION:")
        output.append(f"  Model timestamp: {model_info.get('timestamp', 'N/A')}")
        output.append(f"  Algorithm: {model_info['name']}")
        output.append("")
        output.append("EVALUATION DATASET:")
        output.append(f"  Input file: {input_file}")
        output.append(f"  Dataset size: {sample_size:,} rows")
        output.append("")
        output.append("PERFORMANCE METRICS:")
        output.append(f"  MAE (Mean Absolute Error): {mae:.3f} dB")
        output.append(f"  R² Score: {r2:.3f}")
        output.append("")
        output.append("INTERPRETATION:")
        output.append(f"  Average prediction error: ±{mae:.1f} dB")
        if r2 >= 0:
            output.append(f"  Model explains {r2*100:.1f}% of SNR variance")
        else:
            output.append(f"  Model performs worse than baseline (R²={r2:.3f})")
        output.append("")
        
        if mae < 5:
            quality = "EXCELLENT"
        elif mae < 10:
            quality = "GOOD"
        elif mae < 15:
            quality = "ACCEPTABLE"
        else:
            quality = "NEEDS IMPROVEMENT"
        
        output.append(f"MODEL QUALITY: {quality}")
        output.append("="*70)
        
        results_text = "\n".join(output)
        with open(output_dir / "evaluation_results.txt", "w") as f:
            f.write(results_text)
        
        logger.info(f"Saved: {output_dir / 'evaluation_results.txt'}")
        print("\n" + results_text)
    except Exception as e:
        logger.error(f"ERROR: Failed to save evaluation results: {e}")
        sys.exit(1)

def plot_predictions_vs_actual(y_true, y_pred, output_dir):
    logger.info("Creating predictions vs actual plot...")
    
    try:
        sample_idx = np.random.choice(len(y_true), min(10000, len(y_true)), replace=False)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(y_true[sample_idx], y_pred[sample_idx], alpha=0.3, s=1, c='steelblue')
        
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel("Actual SNR (dB)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Predicted SNR (dB)", fontsize=12, fontweight='bold')
        ax.set_title("Predictions vs Actual SNR", fontsize=14, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(output_dir / "predictions_vs_actual.png", dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_dir / 'predictions_vs_actual.png'}")
        plt.close()
    except Exception as e:
        logger.error(f"ERROR: Failed to create predictions vs actual plot: {e}")
        sys.exit(1)

def plot_residuals(residuals, output_dir):
    logger.info("Creating residuals plot...")
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(residuals, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].set_xlabel("Residual (Actual - Predicted, dB)", fontweight='bold')
        axes[0].set_ylabel("Frequency", fontweight='bold')
        axes[0].set_title("Residuals Distribution", fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        sample_idx = np.random.choice(len(residuals), min(10000, len(residuals)), replace=False)
        axes[1].scatter(range(len(sample_idx)), residuals[sample_idx], alpha=0.3, s=1, c='steelblue')
        axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[1].set_xlabel("Sample Index", fontweight='bold')
        axes[1].set_ylabel("Residual (dB)", fontweight='bold')
        axes[1].set_title("Residuals Scatter Plot", fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "residuals_plot.png", dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_dir / 'residuals_plot.png'}")
        plt.close()
    except Exception as e:
        logger.error(f"ERROR: Failed to create residuals plot: {e}")
        sys.exit(1)

def plot_band_wise_performance(y_true, y_pred, bands, output_dir):
    logger.info("Creating band-wise performance plot...")
    
    try:
        df_eval = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred,
            'band': bands
        })
        
        band_mae = df_eval.groupby('band').apply(
            lambda x: mean_absolute_error(x['actual'], x['predicted']),
            include_groups=False
        ).sort_values()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(band_mae)), band_mae.values, color='steelblue', edgecolor='black')
        ax.set_xticks(range(len(band_mae)))
        ax.set_xticklabels(band_mae.index, rotation=45)
        ax.set_xlabel("Band", fontsize=12, fontweight='bold')
        ax.set_ylabel("MAE (dB)", fontsize=12, fontweight='bold')
        ax.set_title("Model Performance by Band", fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, mae_val in zip(bars, band_mae.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mae_val:.2f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / "band_wise_performance.png", dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_dir / 'band_wise_performance.png'}")
        plt.close()
    except Exception as e:
        logger.error(f"ERROR: Failed to create band-wise performance plot: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='WSPR Model Evaluation')
    parser.add_argument('--input', default='data/spots.20.parquet',
                       help='Input parquet file (default: data/spots.20.parquet)')
    parser.add_argument('--model', default=None,
                       help='Model file path (default: latest model in models/)')
    parser.add_argument('--scaler', default=None,
                       help='Scaler file path (default: auto-detect from model timestamp)')
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("WSPR MODEL EVALUATION")
    logger.info("="*70)
    logger.info(f"Evaluation ID: {RUN_TIMESTAMP}")
    logger.info(f"Input: {args.input}")
    logger.info("")
    
    model_data, model_path = load_model(args.model)
    model = model_data["model"]
    X_columns = model_data["X_columns"]
    model_timestamp = model_data.get("timestamp", None)
    
    logger.info(f"Model: {model_data['name']}")
    logger.info(f"Model timestamp: {model_timestamp}")
    logger.info("")
    
    scaler = load_scaler(args.scaler, model_timestamp)
    logger.info("")
    
    output_dir = Path(f"outputs/07_evaluation/run_{RUN_TIMESTAMP}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    input_path = check_data_file(args.input)
    y, y_pred, bands, mae, r2, residuals, total_count = evaluate_model(model, scaler, input_path, X_columns)
    
    save_results(mae, r2, total_count, output_dir, model_data, args.input)
    plot_predictions_vs_actual(y, y_pred, output_dir)
    plot_residuals(residuals, output_dir)
    plot_band_wise_performance(y, y_pred, bands, output_dir)
    
    logger.info("")
    logger.info("="*70)
    logger.info("MODEL EVALUATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Evaluation ID: {RUN_TIMESTAMP}")
    logger.info(f"Evaluated on {total_count:,} spots")
    logger.info(f"Evaluation MAE: {mae:.3f} dB")
    logger.info(f"Evaluation R²: {r2:.3f}")
    logger.info(f"Plots generated from {len(y):,} sample spots")
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info("="*70)

if __name__ == "__main__":
    main()
