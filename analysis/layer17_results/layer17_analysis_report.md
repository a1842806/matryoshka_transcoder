# Layer 17 Matryoshka Transcoder Analysis Report

**Generated on:** 2025-10-13 13:22:35

## Executive Summary

- **Total Features:** 4,608
- **Dead Features:** 0 (0.0%)
- **Features with Samples:** 4,505
- **Total Samples Collected:** 299,448
- **Average Samples per Feature:** 66.5

## Training Configuration

- **Model:** Gemma-2-2B
- **Layer:** 17 (resid_mid → mlp_out transcoder)
- **Dictionary Size:** 4,608 (Matryoshka: [1152, 2304, 1152])
- **Top-k:** 48
- **Learning Rate:** 0.0003 with warmup+decay
- **Training Steps:** ~14,500

## Model Weight Analysis

### Encoder Weights (W_enc)
- **Mean:** 0.0004
- **Std:** 0.0284
- **Range:** [-0.1250, 0.1250]

### Decoder Weights (W_dec)
- **Mean:** 0.0000
- **Std:** 0.0209
- **Range:** [-0.3789, 0.6719]

### Feature Norms
- **Mean:** 1.3438
- **Std:** 0.2383
- **Range:** [0.9492, 2.6406]

## Top Features by Activation

| Rank | Feature ID | Max Activation | Sample Count |
|------|------------|----------------|--------------|
|  1 |       1103 |          44.25 |          100 |
|  2 |        568 |          36.00 |          100 |
|  3 |       1016 |          35.50 |           23 |
|  4 |        426 |          35.00 |          100 |
|  5 |       1791 |          34.75 |           81 |
|  6 |        629 |          34.75 |          100 |
|  7 |       2319 |          34.00 |            4 |
|  8 |         67 |          33.00 |          100 |
|  9 |        486 |          32.50 |          100 |
| 10 |        494 |          31.88 |          100 |
| 11 |       2994 |          31.88 |           39 |
| 12 |       1089 |          31.50 |           25 |
| 13 |        457 |          31.25 |          100 |
| 14 |       3025 |          31.12 |          100 |
| 15 |         14 |          31.00 |          100 |
| 16 |       3369 |          30.75 |           22 |
| 17 |       2433 |          30.62 |           25 |
| 18 |        393 |          30.50 |          100 |
| 19 |        247 |          30.25 |           46 |
| 20 |       2172 |          30.25 |           39 |

## Feature Analysis

### High-Activation Features

**Feature 1103:**
- Max activation: 44.25
- Sample count: 100
- Top tokens: 235248(5), 235303(3), 235269(2), 235274(2), 9563(1)

**Feature 568:**
- Max activation: 36.00
- Sample count: 100
- Top tokens: 578(4), 235265(3), 235269(3), 6050(2), 1536(1)

**Feature 1016:**
- Max activation: 35.50
- Sample count: 23
- Top tokens: 235276(13), 235284(4), 235274(3)

**Feature 426:**
- Max activation: 35.00
- Sample count: 100
- Top tokens: 603(7), 1841(5), 2439(5), 708(3)

**Feature 1791:**
- Max activation: 34.75
- Sample count: 81
- Top tokens: 2142(6), 235256(3), 235269(2), 235349(2), 235303(2)

## Key Insights

1. **Feature Utilization:** Good feature utilization with low dead feature rate.
2. **Activation Distribution:** Strong feature activations detected.
3. **Sample Collection:** Good sample coverage across features.

## Recommendations

1. **Further Analysis:** Investigate the most active features for interpretability.
2. **Feature Engineering:** Consider adjusting dictionary size based on dead feature rate.
3. **Training:** Monitor convergence - training appears successful.
4. **Interpretability:** Use activation samples for feature interpretation.

## Files Generated

- `layer17_analysis_overview.png`: Overview plots
- `activation_vs_samples.png`: Activation vs sample count analysis
- `layer17_analysis_report.md`: This report
