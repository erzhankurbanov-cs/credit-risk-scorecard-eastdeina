# credit-risk-scorecard-eastdeina
Yerzhan Kurbanov. Financial Risk Management &amp; Data Science. Credit risk scorecard model: Logistic Regression with PDO + LightGBM (WoE &amp; raw features). Full MNK assumption check, Optuna hyperparameter tuning, SHAP, PSI monitoring, PD-segmentation A1-D3.
# Credit Risk Scorecard - Retail Loan Default Modeling

End-to-end credit risk modeling pipeline for retail lending portfolio.
Three-model architecture: Logistic Regression with PDO scorecard + 
LightGBM on WoE-transformed features + LightGBM on raw features.

**Author:** Yerzhan Kurbanov  
**Program:** Financial Risk Management & Data Science  
**Course:** Credit Risk Management

---

## Project Overview

The objective is to build a production-grade scoring solution for a retail
loan portfolio of 1,000,000 borrowers (5% default rate). The pipeline 
follows industry standards (Siddiqi "Intelligent Credit Scoring", Hosmer 
& Lemeshow "Applied Logistic Regression", Basel II/III IRB approach, 
BCBS 239, IFRS 9 ECL).

### Models

| Model | Features | Test Gini | Test KS | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | WoE-transformed (21 vars) | 0.7331 | 0.5478 | 0.4106 |
| LightGBM | WoE-transformed (21 vars) | 0.7642 | 0.5777 | 0.4663 |
| **LightGBM (Champion)** | **Raw features (23 vars)** | **0.8149** | **0.6132** | **0.5421** |

All models exceed the industry "good model" threshold (Gini >= 40%) by 
33-41 percentage points.

---

## Methodology

### 1. Exploratory Data Analysis
- Distribution analysis with log-transform for monetary features
- Pearson correlation matrix with mutual coefficient analysis
- Default rate analysis by feature quartiles

### 2. Train/Test Split
- Stratified 70/30 split on full 1M dataset (random_state=42)
- 700,000 training observations, 300,000 test observations
- Identical default rate (5.0%) in both samples confirmed stratification

### 3. WoE Binning with Forced Monotonicity
- Initial binning via scorecardpy tree method
- Iterative refit for non-monotonic variables (reducing bin_num_limit 
  until monotonicity achieved)
- IV-based feature selection (threshold 0.02)

### 4. Logistic Regression - Full Assumption Check (7 tests)
1. Logit linearity (corr WoE vs logit > 0.95): 23/23 variables passed
2. Multicollinearity (VIF < 5): removed V6, V7 - 21 variables retained
3. Pairwise correlation (|r| < 0.7): 7 pairs flagged, compensated by VIF
4. Sample size (10 EPV rule): EPV = 1,666 (166x minimum requirement)
5. Cook's distance (D < 1): max D = 0.0036, no influential outliers
6. Hosmer-Lemeshow goodness-of-fit
7. Durbin-Watson autocorrelation test

### 5. PDO Scorecard
- Base score: 600 (industry standard FICO/Experian)
- PDO: 20 (every 20 points doubles good odds)
- Base odds: 1/19 (corresponds to 5% bad rate)
- Score range: 338-717

### 6. LightGBM Hyperparameter Tuning
- Optuna with TPE sampler, 25 trials
- 3-fold stratified cross-validation, ROC-AUC objective
- Class imbalance via scale_pos_weight (~19)

### 7. Model Validation
- ROC, Precision-Recall, CAP curves
- KS-statistic, AUC train/test gap analysis
- Reliability diagram (calibration curve)
- Confusion matrix at threshold 0.5

### 8. SHAP Interpretation
- Global feature importance (mean |SHAP|)
- Comparison with Information Value rankings (avg rank diff = 6.09)
- Waterfall plots for individual borrowers

### 9. Population Stability Index
- Per-model PSI on score distributions
- Per-variable PSI (threshold 0.15)
- All variables stable

### 10. Champion-Challenger Comparison
- Full metrics comparison across three models
- Production deployment recommendations

### 11. PD Segmentation A1-D3
- Mapping predicted PD to risk grades per reference scale
- Observed vs expected default rate by segment
- Portfolio concentration analysis

---

## Tech Stack

- **Python 3.11**
- **Data**: pandas, numpy
- **Modeling**: scikit-learn, lightgbm, scorecardpy, statsmodels
- **Hyperparameter tuning**: optuna
- **Interpretation**: shap
- **Visualization**: matplotlib, seaborn
- **Reporting**: python-docx (custom Word export with TNR formatting)
- **Environment**: Kaggle (30 GB RAM, full 1M dataset, no sampling)

---

## Repository Structure

- `srm1-credit-scorecard-v2.ipynb` - main analysis notebook
- `СРМ1_Курбанов_финал.docx` - formatted Word report (Russian)
- `README.md` - this file

The dataset (`data.csv`, 1M observations, 24 columns) is private and 
not included in the repository. Available on request for academic 
verification.

---

## Key Findings

1. LightGBM on raw features outperforms LR on WoE by 8.18 p.p. Gini, 
   driven by the model's ability to capture non-linear feature 
   interactions that one-dimensional WoE binning loses.
2. IV and SHAP rankings show strong alignment (avg rank diff ~6 of 21), 
   confirming both models rely on the same risk drivers; SHAP 
   additionally captures non-linear contributions invisible to IV.
3. PSI of all model scores is effectively 0, confirming distribution 
   stability between train and test (expected for random split, 
   establishes baseline for production OOT monitoring).
4. Portfolio concentration: 56.19% in low-risk grades A1-B3, 43.81% in 
   higher-risk grades C1-D3. Recommends careful cut-off calibration in 
   production based on bank risk appetite.

---

## Production Recommendation

**Multi-model architecture** for deployment:

1. **Primary (Production)**: LR on WoE with PDO scorecard - first-line 
   decisions, regulatory reporting (IFRS 9), full explainability for 
   adverse action notices.
2. **Challenger (Shadow-mode)**: LightGBM on raw - parallel scoring, 
   accumulates evidence for A/B comparison; required SHAP explanations 
   for regulatory compliance if promoted.
3. **Reserve**: LightGBM on WoE - intermediate option for situations 
   requiring higher accuracy than LR while maintaining standard WoE 
   audit trail.

**Monitoring stack:**
- Score PSI: monthly, alert threshold 0.15
- Per-variable PSI: monthly, alert threshold 0.15
- Out-of-time Gini: quarterly, alert on >5 p.p. drop
- Brier score and PD calibration by segment: quarterly
- Full retraining: triggered by alerts or annually minimum

---

## License

MIT License - see LICENSE file.

## Contact

LinkedIn: [https://www.linkedin.com/in/yerzhan-kurbanov-a580b51b1/]  
Email: [er_kurbanov@kbtu.kz]
