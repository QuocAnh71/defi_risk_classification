# Decentralized Credit Scoring: Quantifying Liquidation Risks on Ethereum Lending Markets

### Project Metadata & Team Context

**Course:** Final Project - Machine Learning I

**Institution:** Eastern International University (EIU) - Becamex Business School

**Team Name:** Totoro

**Authors:**

1. **Đặng Quốc Anh**
2. **Nguyễn Hoài Thu**
3. **Lê Nhật Trường**
4. **Lê Hoàng Lâm**

**Primary Data Source:** [Spectral Labs Credit Scoring Training Dataset](https://huggingface.co/datasets/spectrallabs/credit-scoring-training-dataset)

**Target Definition:** Binary indicator representing the Liquidation Risk (`target = 1`) of a borrower's wallet address across major Ethereum lending protocols (Aave v2 and Compound v2).

---

## Repository Tree Architecture

The project codebase is engineered defensively using strict MLOps standards to facilitate reproducible, institutional-grade credit risk auditing:

* `data/`: Segregated file store partitioning static immutable stages (`raw/dataset.parquet` and optimized `processed/` arrays).
* `models/`: Production-ready directory housing frozen, serialized model binaries (`.joblib`).
* `notebooks/`: Synchronous execution engine spanning sequentially from Phase 1 (EDA) to Phase 4 (Estimation & Explainability).
* `reports/figures/`: Materialized storage for 19 high-resolution analytical plots, evaluation curves, and confusion heatmaps completely insulated from graphics memory leaks.
* `src/`: Modular backend package housing declarative system drivers (`preprocessing.py`, `features.py`, `train.py`).

---

## 1. BUSINESS UNDERSTANDING

### Problem Statement

The foundational fragility of over-collateralized crypto loans lies in their dependency on highly volatile asset prices. When the value of deposited collateral drops sharply, liquidations are triggered programmatically via smart contracts. These forced liquidations often trigger cascading market failures across decentralized protocols during extreme drawdowns, creating severe systemic liquidity crunches and amplifying market panic.

### Business Importance

Proactive, data-driven credit risk modeling is critical to protecting protocol capital reserves and preserving the safety of liquidity providers. By identifying and isolating high-risk wallets before market corrections materialize, decentralized lending markets can proactively mitigate insolvency events, curb bad debt accumulation, and stabilize the broader DeFi ecosystem.

### Stakeholders

* **DeFi Lending Markets (Aave/Compound Governance Councils):** To dynamically optimize liquidation thresholds, reserve factors, and borrow bounds.
* **Decentralized Asset Managers:** To shield user principal allocations from black-swan liquidations and optimize capital utilization.
* **Yield Aggregators:** To intelligently route cross-protocol liquidity using rigorous risk-adjusted return matrices.

### Project Objective

The objective is formulated as a highly imbalanced binary classification task predicting whether a borrower's wallet will cross its critical solvency boundary and face liquidation. The quantitative system must optimize the trade-off between maximizing default tracking capacity (Recall) to avoid protocol losses, while maintaining strict classification Precision to prevent the false rejection of creditworthy market participants.

---

## 2. DATA UNDERSTANDING

### Dataset Introduction

The raw historical ledger captures the comprehensive on-chain behavioral footprint of borrowers. After applying strict temporal partitioning to guarantee absolute out-of-sample isolation, the data matrices span:

* **Training Partition:** 402,754 observations × 15 finely curated orthogonal features.
* **Out-of-Time (OOT) Holdout Test Partition:** 40,207 observations × 15 finely curated orthogonal features.
* **Target Imbalance:** The empirical distribution exhibits a tight ~12%+ default baseline density, imposing severe class imbalance constraints on standard classifiers.

### Exploratory Data Analysis (EDA)

A rigorous 6-phase analytical funnel was executed to map behavioral signals, with plots committed directly to `reports/figures/`:

1. **Target Distribution Audit:** Quantifying and mapping the class skewness constraints.
2. **Transaction Activities:** Dissecting smart contract call patterns and execution frequencies.
3. **Balance Extremes:** Auditing and normalizing heavily skewed, heavy-tailed token balance spreads.
4. **Micro Risk-Cliffs Gapping:** Mapping non-linear inflection points in wallet health indices.
5. **Macro Market Regime Correlations:** Assessing credit variance anomalies against broader cross-sectional market conditions.
6. **MI/KS Feature Shortlist:** Ranking raw information-theoretic signals via Mutual Information and Kolmogorov-Smirnov statistics.

### Key Insights

On-chain analysis revealed that the micro behavioral vector `max_risk_factor` exhibits a sharp, highly non-linear risk cliff exactly at the **0.7** threshold boundary. Concurrently, macro market indicators suffer from extensive Spearman multicollinearity, dictating defensive regularization strategies in subsequent pipeline layers.

---

## 3. DATA PREPARATION

### Data Cleaning

Implemented modularly within `src/preprocessing.py`, the data-cleaning pipeline utilizes training-bound dynamic percentile capping and clamping. This neutralizes extreme Whale wallet anomalies without corrupting or discarding critical underlying rows. Heavily skewed financial feature spaces are smoothed via native Polars `log1p` mathematical operators, coupled with a strict `np.nan_to_num` defense layer to clean missing cross-protocol data fields uniformly.

### Feature Engineering and Feature Encoding

The infrastructure projects the baseline features into a high-dimensional manifold of **123 engineered features**, mapping multi-dimensional behavioral risks:

* **Gas Economics:** Historical gas consumption indicating transaction urgency and user willingness to pay block premium scalars.
* **Multi-Protocol Diversity Counts:** Ingestion of unique borrowing and lending dApp breadth to measure systemic diversification.
* **Net Cashflow Waterfalls:** Multi-horizon aggregations of ETH token inflows minus outflows.
* **Directional Transaction Velocity:** The trajectory and velocity of capital movement away from the wallet address.

This expansive matrix is subsequently driven through the information-theoretic selection funnel (MI and KS scoring) and strict Spearman multicollinearity pruning ($|r| > 0.80$) to isolate exactly **15 highly orthogonal surviving features** for model training.

### Train-Test Split

To guarantee absolute statistical isolation and prevent future data leakage, the project implements a rigid **Out-of-Time (OOT) Chronological Split** at Unix timestamp `1672531200` (2023-01-01 00:00:00 UTC). The training set isolates bull-market and historical distributions prior to 2023, whereas the test set evaluates model resilience under stress strictly against unseen, out-of-sample 2023 Crypto Winter market drawdowns.

---

## 4. MODEL DEVELOPMENT

### Candidate Models

The system evaluates two distinct tree-based ensemble paradigms tailored for highly non-linear credit risk surfaces:

* **Random Forest Classifier (Bagging Paradigm):** Selected for its robust variance reduction properties. It excels at parsing raw financial noise and complex feature interactions without extending out-of-sample overfitting boundaries.
* **XGBoost Classifier (Boosting Paradigm):** Chosen for its premium bias reduction capabilities. It excels at optimizing custom loss functions and handling data sparsity natively via specialized *Sparsity-aware Split Finding* routines.

### Model Training

Inside the `src/train.py` driver, automated hyperparameter optimization is orchestrated using the **Optuna** framework. Cutoff thresholds are determined by executing a continuous Macro F1-Score search sweeping from `0.1` to `0.9` **strictly in-sample on the training partition** to safely freeze the operational boundaries (Frozen Thresholds). To neutralize residual Spearman multicollinearity, the XGBoost engine is injected with rigid L1/L2 structural regularization parameters (`reg_alpha` and `reg_lambda`).

---

## 5. MODEL EVALUATION

### Model Comparison Table

Out-of-sample performance evaluation metrics generated after locking and freezing the operational classification thresholds derived natively from the training split:

| Model Architecture         | ROC-AUC      | Gini Coefficient | PR-AUC       | Optimal Threshold (Frozen) | Accuracy     | Precision    | Recall       | F1-Score     |
| -------------------------- | ------------ | ---------------- | ------------ | -------------------------- | ------------ | ------------ | ------------ | ------------ |
| **Random Forest Baseline** | 0.912524     | 0.825049         | 0.837491     | 0.41                       | 0.823563     | 0.629619     | **0.833117** | 0.717213     |
| **XGBoost Champion**       | **0.917128** | **0.834255**     | **0.846024** | **0.44**                   | **0.839132** | **0.666232** | 0.803575     | **0.728486** |

### Visual Evaluation

Our dual-panel diagnostic chart located at `reports/figures/model_evaluation_curves.png` plots the out-of-sample ROC Curve (Left) side-by-side with the Precision-Recall Curve (Right). The PR Curve demonstrates XGBoost's superior precision lift over Random Forest across highly imbalanced regions. Concurrently, `confusion_matrices.png` maps exact wallet distribution counts within spatial quadrants locked strictly to the frozen train thresholds, ensuring zero test-set contamination.

### Best Model Selection & Justification

**The XGBoost Classifier is signed off as the definitive Champion Model**. This choice is backed by rigorous quantitative risk parameters: XGBoost achieves dominant global metrics across **ROC-AUC (91.71%)**, **Gini (83.43%)**, **PR-AUC (84.60%)**, and secures the highest macro **F1-Score (0.7285)**.

While Random Forest captures a slightly broader default pool (+2.95% Recall) by lowering its cutoff to `0.41`, it suffers an severe Precision penalty (dropping to 62.96% compared to XGBoost's 66.62%). In decentralized credit underwriting, XGBoost's +3.66% precision advantage prevents multi-million dollar capital rejections of healthy wallets, preserving protocol transaction volume while maintaining a premier default detection rate (Recall) of **80.36%**.

---

## 6. MODEL INTERPRETATION

### Final Model Interpretation

To dismantle the "black-box" nature of the boosting ensemble, the project integrates game-theoretic explainability utilizing **SHAP (TreeExplainer)** to isolate exact Shapley feature contributions. The computation pass forces an explicit `check_additivity=False` override to neutralize micro-precision floating-point convergence dust ($10^{-6}$ deltas) from impeding the pipeline execution.

The global summary plot at `reports/figures/shap_summary_plot.png` confirms that the champion model is driven by financially sound underwriting rules:

1. **`LTV_Utilization_Quotient_log1p`:** Serving as the primary global risk vector, elevated loan-to-value proxies scale risk profiles non-linearly, pushing wallets toward liquidation thresholds.
2. **`risk_factor_above_threshold_daily_count`:** High frequency act as a compounding behavioral penalty layer, validating sustained risk exposure over time rather than isolated, transient spikes.

---

## 7. BUSINESS RECOMMENDATIONS AND LIMITATIONS

### Actionable Recommendations

1. **Dynamic Interest Rate Curves:** Lending protocols should pivot toward dynamic borrowing rates directly indexed to wallet-level SHAP risk parameters, algorithmically penalizing high-risk capital configurations.
2. **Pre-emptive Liquidation Warnings:** Implement on-chain or off-chain oracle warnings triggered the moment a wallet approaches the optimal F1 threshold (`0.44`) or breaches the `0.7` risk factor cliff, incentivizing automated or manual self-repayment before bad debt materializes.

### Limitations

* **Single-Chain Ingestion Bottleneck:** The data ingestion architecture is bounded strictly by Ethereum Mainnet history. It does not account for cross-chain margin positions or layer-2 collateralization vectors.
* **Hardware Power Constraints:** Computation boundary constraints and hardware thermal throttling narrowed the sweeping resolution of our Optuna hyperparameter grid search.

### Future Work

* **Cross-Chain Contagion Modeling:** Expanding the Polars graph ingestion layer to scan multi-chain credit contagion across rapidly scaling L2 rollups (Arbitrum, Optimism, Base).
* **Real-Time Streaming Pipelines:** Transitioning batch analytics into continuous, real-time risk assessment engines leveraging the massive throughput of Polars LazyFrame nodes.