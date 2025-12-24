# WSPR Band Recommendation & SNR Regression

![Status](https://img.shields.io/badge/Status-Training_Complete-orange)
![Model](https://img.shields.io/badge/Model-Failed_(R²=-0.009)-red)
![Data](https://img.shields.io/badge/Dataset-417M_rows-blue)

**Machine Learning project for predicting Signal-to-Noise Ratio (SNR) in WSPR (Weak Signal Propagation Reporter) amateur radio transmissions.**

⚠️ **Project Status**: Model training completed but failed to achieve acceptable performance. This repository documents the process, challenges, and lessons learned. See [RESULTS.md](RESULTS.md) for detailed analysis.

---

## Project Overview

This project analyzes **417 million** WSPR spot records (20% stratified sample) to build a predictive model for signal strength (SNR) based on:
- **Propagation distance** between transmitter and receiver
- **Time of day** and **season** (cyclic features)
- **Frequency band** (LF to VHF)

**Project Status**: Model training completed but failed to achieve acceptable performance (R² = -0.009). See `RESULTS.md` for detailed analysis and lessons learned.

---

## Dataset

- **Source**: [wspr.live](https://wspr.live) ClickHouse database
- **Acquisition**: SQL queries via `00_data_acquisition.py` (~18 hours)
- **Full Dataset**: 2,086,489,757 spots (June 2024 - June 2025)
- **Working Dataset**: 417,297,944 spots (20% stratified sample by band)
- **Features**: Time, Band, Distance, SNR, TX/RX coordinates
- **Format**: Parquet (partitioned for memory-efficient processing)

### Data Acquisition Method

Data fetched successfully using direct ClickHouse SQL queries to wspr.live database backend. Implemented parallel batch downloading (4 workers, 6-hour batches) with caching and resume capability. Total download time: ~18 hours for 2B rows. See `RESULTS.md` Section 2 for technical details on the SQL-based approach.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `pyarrow`

### 2. Run Scripts

```bash
# Create sample from full dataset
python src/01_create_samples.py --input data/spots.full.parquet --percent 20

# Data analysis
python src/02_data_overview.py --input data/spots.20.parquet
python src/03_data_quality.py --input data/spots.20.parquet
python src/04_band_snr_analysis.py --input data/spots.20.parquet
python src/05_distance_correlation.py --input data/spots.20.parquet

# Model training and evaluation
python src/06_train_model.py
python src/07_evaluate_model.py
```

---

## Analysis Scripts

### 📊 02_data_overview.py
**Basic statistics and band distribution**

Outputs:
- `outputs/02_overview/statistics.txt` - Dataset summary
- `outputs/02_overview/band_distribution.png` - Spot counts by band
- `outputs/02_overview/band_counts.csv` - Band statistics

**Key Findings**:
- Most active bands: **20m (29.5%)**, **40m (26.9%)**, **30m (16.3%)**
- SNR range: -99 to +100 dB
- Average distance: 2,103 km

---

### 🔍 03_data_quality.py
**Data quality checks and validation**

Outputs:
- `outputs/03_quality/quality_report.txt` - Quality metrics
- `outputs/03_quality/outlier_analysis.png` - SNR distribution with outliers

**Quality Results**:
- **Missing values**: 0
- **Invalid data**: <0.001%
- **Data quality**: EXCELLENT ✅

---

### 📡 04_band_snr_analysis.py
**SNR analysis across frequency bands**

Outputs:
- `outputs/04_band_snr/band_statistics.csv` - Band-wise SNR stats
- `outputs/04_band_snr/band_snr_boxplot.png` - SNR distribution comparison
- `outputs/04_band_snr/snr_histogram_by_band.png` - Detailed histograms

**Key Insights**:
- Higher bands (10m-20m) show **wider SNR variation**
- Lower bands (40m-80m) more **consistent** but weaker signals
- High overlap between bands indicates prediction difficulty

---

### 📏 05_distance_correlation.py
**Distance vs SNR relationship analysis**

Outputs:
- `outputs/05_distance/correlation_matrix.png` - Feature correlations
- `outputs/05_distance/distance_vs_snr_scatter.png` - Scatter plot
- `outputs/05_distance/distance_vs_snr_hexbin.png` - Density visualization

**Key Findings**:
- **Weak correlation** between distance and SNR (r = -0.25)
- Signal strength decreases with distance but with high variance
- Features lack strong predictive power

---

### 🤖 06_train_model.py
**Model training**

Outputs:
- `models/best_model_*.pkl` - Trained SGDRegressor
- `models/scaler_*.pkl` - Feature scaler
- `outputs/06_training/run_*/training_results.txt` - Training metrics

**Model Configuration**:
- **Algorithm**: SGDRegressor (Stochastic Gradient Descent)
- **Penalty**: L2 (Ridge regularization)
- **Training data**: 312M rows (75% of 20% sample)
- **Features**: 18 (distance, cyclic time, one-hot band)

**Why SGDRegressor?**
- ✅ **Incremental learning** - handles massive datasets
- ✅ **Memory efficient** - mini-batch processing
- ✅ **Fast training** - ~6 minutes

**Performance**: MAE=7.3 dB, R²=-0.009 (failed)

---

### 📈 07_evaluate_model.py
**Model evaluation and visualization**

Outputs:
- `outputs/07_evaluation/run_*/evaluation_results.txt` - Performance metrics
- `outputs/07_evaluation/run_*/predictions_vs_actual.png` - Prediction quality
- `outputs/07_evaluation/run_*/residuals_plot.png` - Error distribution
- `outputs/07_evaluation/run_*/band_wise_performance.png` - Per-band accuracy

**Evaluation Metrics**:
- **MAE (Mean Absolute Error)**: Average prediction error in dB
- **R² Score**: Model quality vs baseline
- **Band-wise MAE**: Performance across different bands

---

## Memory Management

All scripts are **memory-safe** for handling 400+ million rows with 32GB RAM:

**Strategies**:
- **Partition-by-partition processing** - reads data in chunks
- **Sampling** for visualizations when needed
- **Mini-batch training** - 1M rows per batch
- **Incremental metrics** - no large list accumulations
- **Explicit garbage collection** - `gc.collect()` after heavy operations

---

## Pipeline Overview (8 Scripts)

| Script | Purpose | Status | Key Output |
|--------|---------|--------|------------|
| `00_data_acquisition.py` | Fetch data via ClickHouse SQL | ✅ Success | 2B rows (~18 hours) |
| `01_create_samples.py` | Create stratified sample | ✅ Success | 417M rows (20% sample) |
| `02_data_overview.py` | Basic statistics | ✅ Success | Mean SNR: -14.86 dB |
| `03_data_quality.py` | Quality validation | ✅ Success | 99.9993% clean data |
| `04_band_snr_analysis.py` | Band-wise SNR analysis | ✅ Success | High overlap detected |
| `05_distance_correlation.py` | Feature correlations | ✅ Success | Weak correlations (r<0.3) |
| `06_train_model.py` | Train SGDRegressor | ✅ Success | Training complete |
| `07_evaluate_model.py` | Model evaluation | ✅ Success | R²=-0.009 (worse than baseline) |

**See [RESULTS.md](RESULTS.md) for detailed analysis.**

---

## Project Structure

```
wspr-band-recommendation-snr-regression/
├── data/
│   ├── spots.full.parquet/             # 2B rows (65GB, not in repo)
│   └── spots.20.parquet/               # 417M rows (13GB, not in repo)
├── src/
│   ├── 00_data_acquisition.py          # Data fetching via SQL
│   ├── 01_create_samples.py            # Stratified sampling
│   ├── 02_data_overview.py             # Basic statistics
│   ├── 03_data_quality.py              # Quality validation
│   ├── 04_band_snr_analysis.py         # SNR analysis by band
│   ├── 05_distance_correlation.py      # Correlation analysis
│   ├── 06_train_model.py               # Model training
│   └── 07_evaluate_model.py            # Model evaluation
├── outputs/
│   ├── 02_overview/                    # Statistics + plots
│   ├── 03_quality/                     # Quality reports
│   ├── 04_band_snr/                    # SNR analysis plots
│   ├── 05_distance/                    # Correlation plots
│   ├── 06_training/                    # Training logs
│   └── 07_evaluation/                  # Evaluation results + plots
├── models/
│   ├── best_model_20251220_095418.pkl  # Trained model
│   └── scaler_20251220_095418.pkl      # Feature scaler
├── logs/                               # Script execution logs
├── README.md                           # This file
├── RESULTS.md                          # Detailed project report
└── requirements.txt                    # Dependencies
```

---

## Results Summary

### Dataset Statistics
- **Total Spots**: 2,086,489,757
- **Bands**: 13 (LF to 6m)
- **Distance Range**: 0 - 20,010 km
- **SNR Range**: -99 to +100 dB

### Data Quality
- **Missing Values**: 0
- **Invalid Data**: <0.001%
- **Outliers**: Kept (represent real propagation conditions)

### Model Performance

- **Algorithm**: SGDRegressor with mini-batch learning
- **Training Data**: 312,973,454 rows (75% of 20% sample)
- **Evaluation Data**: 417,297,944 rows (full 20% sample)
- **MAE**: 7.300 dB
- **R² Score**: -0.009 (model failed to learn meaningful patterns)
- **Training Time**: 6.1 minutes
- **Status**: **UNSUCCESSFUL** - Model performs worse than baseline

**See `RESULTS.md` for detailed analysis of failure reasons and lessons learned.**

---

## Technical Notes

### Hardware Requirements
- **RAM**: 32 GB (project tested with this configuration)
- **Storage**: 65+ GB for full dataset, 15GB for 20% sample
- **CPU**: Multi-core recommended for parallel processing
- **Warning**: Lower RAM configurations may cause system instability (BSOD on Windows)

### Performance Tips
1. **Use SSD** for data storage (faster I/O)
2. **Close other applications** during training
3. **Run scripts individually** if memory-constrained
4. **Use `--skip-training`** for repeated analysis

### Troubleshooting

**Memory Issues**:
```bash
# Reduce sample size in scripts
frac=0.01  →  frac=0.005  # Use smaller sample
```

**Slow Training**:
```bash
# Training uses 20% sample - already optimized
# Consider using fewer partitions or smaller sample
```

---

## Critical Issues and Future Directions

### Current Status: Model Failed

The current SGDRegressor model achieved **R² = -0.009** (negative), indicating it failed to learn meaningful patterns. See `RESULTS.md` for detailed analysis.

### Required Improvements for Success

**Priority 1: Add Essential Features**
- [ ] Solar flux data (F10.7 index) from NOAA
- [ ] Geomagnetic K-index
- [ ] Ionospheric MUF predictions
- [ ] Sunrise/sunset times at TX/RX locations

**Priority 2: Algorithm Changes**
- [ ] Switch to XGBoost or Random Forest (requires more RAM)
- [ ] Consider classification instead of regression ("Good"/"Fair"/"Poor")
- [ ] Implement ensemble methods

**Priority 3: Data Quality**
- [ ] Reduce dataset size, increase feature quality
- [ ] Filter out known noisy conditions
- [ ] Use temporal sequences (LSTM) instead of independent samples

**Optional Enhancements**
- [ ] Real-time band recommendation API
- [ ] Web dashboard for visualization
- [ ] Integration with live WSPR data

---

## License

This project is for educational purposes. WSPR data courtesy of [wspr.live](https://wspr.live).

---

## Author

Created for Machine Learning course project.

**Contact**: [Your contact info]

---

## Acknowledgments

- **WSPR Community** for open data
- **wspr.live** for ClickHouse database access
- **Scikit-learn** for ML framework

---

**Project Status**: Training complete - Model failed (R²=-0.009)  
**For detailed analysis**: See [RESULTS.md](RESULTS.md)
