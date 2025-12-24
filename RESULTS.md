<div align="center">

<br><br>

# WSPR SNR PREDICTION
## Project Results and Analysis

<br>

### Machine Learning Course Project

<br>

**Author**  
M. Furkan Eminer  
25040201008

<br>

**Instructor**  
Doç Dr Önder Çoban

<br>

---

**Dataset**

2,086,489,757 Total WSPR Spots (June 2024 - June 2025)  
417,297,944 Spots Analyzed (20% Stratified Sample)

---

**Source Code**

https://github.com/mfeminer/wspr-band-recommendation-snr-regression

---

<br>

December 2025

<br><br>
<br><br>
<br><br>
<br><br>
<br><br>

</div>

---

## Executive Summary

This project attempted to predict Signal-to-Noise Ratio (SNR) in amateur radio WSPR transmissions using machine learning. Despite successfully processing 417 million data points and completing model training, the final model failed to achieve acceptable performance:

- **Mean Absolute Error (MAE)**: 7.300 dB
- **R² Score**: -0.009 (negative indicates worse than baseline)
- **Conclusion**: Model did not learn meaningful patterns from the data

This report documents the technical approach, challenges encountered, and lessons learned from this unsuccessful but instructive attempt.

---

## 1. Introduction to WSPR Data

### What is WSPR?

WSPR (Weak Signal Propagation Reporter) is an amateur radio protocol designed to test propagation paths on HF and VHF bands. Amateur radio operators worldwide run automated WSPR transmitters and receivers, creating a global network that monitors radio wave propagation conditions.

### Data Source and Purpose

**Source**: [wspr.live](https://wspr.live) archives (June 2024 - June 2025)  
**Original Purpose**: Real-time propagation monitoring for amateur radio community  
**Our Purpose**: Predict signal strength (SNR) based on distance, time, and frequency band

Each WSPR "spot" represents a successful reception and contains:
- Transmitter and receiver callsigns and coordinates
- Frequency band (numeric code, see table below)
- Signal-to-Noise Ratio (SNR) in dB
- Timestamp (UTC)
- TX/RX coordinates (latitude/longitude)
- Propagation distance (calculated from coordinates)

### WSPR Frequency Bands

| Band Code | Band Name | Frequency Range | Wavelength |
|-----------|-----------|-----------------|------------|
| -1 | LF | 136-137 kHz | ~2200m |
| 0 | MF | 472-479 kHz | ~630m |
| 1 | 160m | 1.8-2.0 MHz | 160 meters |
| 3 | 80m | 3.5-4.0 MHz | 80 meters |
| 5 | 60m | 5.3-5.4 MHz | 60 meters |
| 7 | 40m | 7.0-7.3 MHz | 40 meters |
| 10 | 30m | 10.1-10.15 MHz | 30 meters |
| 14 | 20m | 14.0-14.35 MHz | 20 meters |
| 18 | 17m | 18.068-18.168 MHz | 17 meters |
| 21 | 15m | 21.0-21.45 MHz | 15 meters |
| 24 | 12m | 24.89-24.99 MHz | 12 meters |
| 28 | 10m | 28.0-29.7 MHz | 10 meters |
| 50 | 6m | 50-54 MHz | 6 meters |

WSPR data is commonly used by radio operators for band planning, propagation studies, and antenna performance evaluation.

---

## 2. Data Acquisition

**Method**: Direct ClickHouse SQL queries to wspr.live database via HTTP (`00_data_acquisition.py`)

**Configuration**:
- Batch size: 6 hours per request
- Parallel workers: 4 threads
- Date range: June 2024 - June 2025 (13 months)
- Caching with resume capability

**Results**:
- Total rows: 2,086,489,757
- Download time: ~18 hours
- Output: `data/spots.full.parquet` (65GB)

**Why Parquet**: 3x smaller files (65GB vs 180GB CSV), 20x faster reads, columnar storage enables selective column loading.

### Sampling (`01_create_samples.py`)

**Why Stratified Sampling?** 
Band distribution is highly imbalanced (20m: 29.5%, LF: 0.1%). Random sampling would under-represent rare bands, biasing the model. Stratified sampling ensures each band maintains its original proportion in the 20% sample, preserving dataset representativeness for accurate evaluation.

**Method**: Stratified sampling by frequency band (20% of 2B rows)
**Result**: 417M rows, 13GB file, perfect band distribution preservation (each band exactly 20% of original)

---

## 3. Exploratory Data Analysis

### Data Overview

417M spots, SNR mean: -14.86 dB (σ=9.18), Distance mean: 2,103 km, No missing values. Most active bands: 20m (29.5%), 40m (26.9%), 30m (16.3%).

![Band Distribution](outputs/02_overview/band_distribution.png)
*Figure 1: Band distribution shows extreme imbalance - top 3 bands account for 72.7% of traffic. Rare bands (LF, 6m) <0.2% each. This imbalance justified stratified sampling.*

**Critical Observation**: High SNR standard deviation (±9.18 dB, 62% of mean absolute value) indicated high inherent noise, suggesting prediction difficulty.

### Data Quality

99.9993% valid data (2,729 invalid rows). Statistical outliers (1.22%) retained as they represent real propagation conditions.

![Outlier Analysis](outputs/03_quality/outlier_analysis.png)
*Figure 2: SNR distribution is near-normal with heavy tails. IQR outliers (±20 dB from median) represent extreme propagation conditions (excellent or poor), not data errors. Retaining them prevents model bias toward average conditions.*

### Band SNR Analysis

All bands showed high overlap (σ=9-15 dB) with weak separation. Mean SNR ranged from -9.1 dB (6m) to -19.2 dB (LF). High variance (70-100% of operational SNR range) indicated prediction difficulty.

![Band SNR Box Plot](outputs/04_band_snr/band_snr_boxplot.png)
*Figure 3: Box plots reveal massive overlap - IQR ranges for all bands span -22 to -8 dB. Only 10 dB difference between best (6m) and worst (LF) band means. This weak separation limits band as a discriminative feature.*

![SNR Histogram by Band](outputs/04_band_snr/snr_histogram_by_band.png)
*Figure 4: Histograms for major bands (20m, 40m, 30m, 80m, 15m, 10m) are nearly identical in shape and spread. No band-specific patterns visible. This predicted model's inability to learn band-based rules.*

### Correlation Analysis

Weak correlations across all feature pairs: Distance-SNR (r=-0.25), Band-SNR (r=+0.03), Hour-SNR (r=-0.08). High scatter (±20 dB at same distance) indicated available features had low predictive power.

![Correlation Matrix](outputs/05_distance/correlation_matrix.png)
*Figure 5: Correlation matrix shows all features weakly correlated with SNR (|r| < 0.3). Strongest predictor (distance) explains only 6% of variance (r²=0.0625). This weak correlation structure predicted poor model performance.*

![Distance vs SNR Hexbin](outputs/05_distance/distance_vs_snr_hexbin.png)
*Figure 6: Hexbin density plot reveals no clear distance-SNR trend. At any given distance (e.g., 5,000 km), SNR varies by ±20 dB, indicating distance alone cannot predict signal strength. Unmeasured factors (solar/ionospheric) dominate.*

---

## 4. Feature Engineering

**Final Features** (18 total):
- Distance (km)
- Time: `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos` (cyclic encoding)
- Band: One-hot encoded (13 bands)

**Missing Critical Features**: Solar flux, geomagnetic indices (K-index), ionospheric data (foF2/MUF), transmitter power, antenna characteristics.

---

## 5. Model Selection

**Algorithm**: SGDRegressor (Stochastic Gradient Descent)
**Reason**: Incremental learning via `partial_fit()`, memory-efficient (1M row batches), handles 417M rows with 32GB RAM.
**Alternatives rejected**: XGBoost, Random Forest (require full dataset in RAM), Neural Networks (complex tuning).
**Configuration**: L2 regularization (alpha=0.001), adaptive learning rate, StandardScaler for feature scaling.

---

## 6. Evaluation Metrics

**MAE** (Mean Absolute Error): Average prediction error in dB. Interpretable, robust to outliers. Target: ±2-3 dB. Achieved: 7.3 dB.

**R²** (R-squared): Proportion of variance explained. Values: 1.0 (perfect), 0.0 (baseline mean), <0.0 (worse than baseline). Achieved: -0.009 (model failed).

**Alternative metrics**: RMSE, MAPE, Median Absolute Error.


---

## 7. Training Process

**Memory Challenges**: Initial Dask-based approach caused 3 BSODs (CRITICAL_PROCESS_DIED, WATCHDOG timeouts) due to large partition loading and test data accumulation exceeding 32GB RAM.

**Final Solution**: Abandoned Dask, used pure Pandas with mini-batch processing (1M rows/batch), explicit memory control (`gc.collect()`), peak RAM ~3.5GB.

**Configuration**: 312M training rows (75% of sample), 530 mini-batches, 6.1 minutes total training time.

---

## 8. Results

**Performance**: MAE = 7.3 dB (36% of operational SNR range), R² = -0.009 (worse than baseline mean).

**Interpretation**: Model converged to predicting mean SNR (-15 dB), ignoring input features. Performance equivalent to random guessing.

### Visualizations

![Predictions vs Actual](outputs/07_evaluation/run_20251220_193841/predictions_vs_actual.png)
*Figure 7: Perfect predictions would show diagonal line. Instead, horizontal line at -15 dB proves model ignores inputs and always predicts mean. No learning occurred.*

![Residuals Plot](outputs/07_evaluation/run_20251220_193841/residuals_plot.png)
*Figure 8: Random scatter with ±20 dB spread (133% of operational SNR range). No systematic pattern = model captured no structure. Errors as large as predictions themselves.*

![Band-wise Performance](outputs/07_evaluation/run_20251220_193841/band_wise_performance.png)
*Figure 9: Flat MAE across all bands (6.5-8.0 dB) confirms model treats all frequencies identically. Expected: some bands easier to predict. Reality: uniform failure.*

**EDA predicted failure**: Band overlap (Figure 3-4) + weak correlations (Figure 5-6) → insufficient predictive power → R² ≈ 0.

---

## 9. Failure Analysis

**Root Causes**:

1. **Insufficient Features** (60%): Missing solar flux, geomagnetic indices (K-index), ionospheric data (foF2), transmitter power, antenna characteristics.

2. **High Inherent Noise** (30%): SNR σ=±10-12 dB from multipath fading, interference, atmospheric effects. Even optimal features unlikely to achieve R²>0.5.

3. **Algorithm Limitations** (10%): Linear SGDRegressor inadequate for non-linear ionospheric propagation. XGBoost/Neural Networks infeasible with RAM constraints.

**Successes**: Memory-safe processing (417M rows), stratified sampling, reproducible pipeline, correct EDA interpretation.

---

## 10. Lessons Learned

**Technical**: Feature quality > algorithm complexity. Domain knowledge essential for feature identification. 32GB RAM requires incremental processing, explicit memory control. Weak EDA correlations (r<0.3) correctly predicted failure.

**Process**: Hardware constraints dictate algorithm choice. Conservative RAM usage (<85%) prevents BSODs. Incremental testing (1-5% samples first) validates approach before full-scale runs.

**Project Management**: Dataset size ambition (2B rows) created memory challenges consuming 80% of dev time. Should have pivoted after EDA showed weak correlations.

---

## 11. Conclusion

This project successfully demonstrated large-scale data processing (417M rows), memory-safe pipeline design, robust evaluation methodology, and reproducible workflow. However, the predictive model failed to achieve acceptable performance:

**Performance Metrics:**
- R² = -0.009 (no meaningful learning)
- MAE = 7.3 dB (unusable for practical applications)
- Predictions converge to mean SNR regardless of input features

**Root Cause:** Insufficient features. Radio propagation cannot be predicted from distance, time, and frequency alone. Solar flux indices, geomagnetic activity (K-index), and ionospheric critical frequency (foF2) are essential for accurate SNR prediction.

**Key Learnings:**
- Feature engineering is more critical than algorithm selection
- Domain knowledge is essential for feature identification
- Memory management is crucial for large-scale data processing
- Weak correlations in EDA (r < 0.3) predicted model failure

**Recommendation:** This model should not be deployed for production use. Future work should incorporate solar and ionospheric data or reframe the problem as a classification task (e.g., "Good/Fair/Poor" propagation conditions).

---

## Appendix A: Repository Structure

8 Python scripts (`src/00-07_*.py`), data in Parquet format (`data/spots.*.parquet`), outputs in `outputs/` (plots, statistics), trained models in `models/`. Full structure at https://github.com/mfeminer/wspr-band-recommendation-snr-regression

---

## Appendix B: Hardware

Windows 10, 32GB DDR4 RAM, SSD, CPU-only. Max RAM usage: 85% (27.2 GB) to prevent BSOD.



## References

**Data Source**: [wspr.live](https://wspr.live)  
**WSPR Protocol**: [WSPRnet.org](http://wsprnet.org)  
**Tools Used**: Python, Pandas, Scikit-learn, Parquet (PyArrow)  
**Hardware**: Windows 11, 32GB RAM, SSD storage