# Decentralized Credit Scoring: Quantifying Liquidation Risks on Ethereum Lending Markets

![Python](https://img.shields.io/badge/python-3.14-blue.svg)
![Polars](https://img.shields.io/badge/polars-fast-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ml-yellow.svg)
![XGBoost](https://img.shields.io/badge/xgboost-champion-brightgreen.svg)
![Optuna](https://img.shields.io/badge/optuna-tuning-blueviolet.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

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

Our project is structured as follows to ensure modularity and reproducibility:
- `data/`: Staged data layers (`raw/dataset.parquet` and `processed/` features files).
- `src/`: Modular backend scripts (`preprocessing.py` for cleaning, `features.py` for engineering, `train.py` for modeling).
- `notebooks/`: Sequential execution harness from 1.0 (EDA) to 4.0 (Modeling & SHAP).
- `reports/figures/`: Storage for all phase-specific EDA plots, ROC/PR curves, and SHAP summary charts.

---

## 1. BUSINESS UNDERSTANDING

### Problem Statement
The fundamental fragility of over-collateralized crypto loans lies in their dependence on highly volatile asset prices. When collateral value drops sharply, liquidations are triggered programmatically. These forced liquidations can lead to cascading failures in DeFi protocols during sudden market drawdowns, exacerbating market panic and causing severe liquidity crunches.

### Business Importance
Proactive credit risk modeling matters immensely for liquidity providers and the systemic security of decentralized protocols. By identifying high-risk wallets before market downturns, protocols can mitigate massive capital loss, reduce the probability of bad debt accumulation, and stabilize the broader DeFi ecosystem.

### Stakeholders
- **Lending Protocols (Aave/Compound governance):** To optimize liquidation thresholds and reserve factors.
- **Decentralized Asset Managers:** To protect user funds and optimize yield generation strategies.
- **Yield Aggregators:** To intelligently route liquidity based on risk-adjusted returns.

### Project Objective
The mathematical task is formulated as a highly imbalanced binary classification model predicting whether a wallet will cross its critical health threshold and face liquidation. The objective is to maximize the model's ability to capture potential defaults (Recall) while maintaining high precision.

## 2. DATA UNDERSTANDING

### Dataset Introduction
Our data footprint explicitly details a rich cross-section of decentralized borrowing behavior:
- **Train Set:** 402,754 rows × 53 columns
- **Out-of-Time (OOT) Holdout Test Set:** 40,207 rows × 53 columns
- **Target Imbalance:** The baseline distribution exhibits a tight ~12%+ default imbalance, posing significant modeling challenges.

### Exploratory Data Analysis (EDA)
Our 6-Phase analytical funnel is rigorously documented and saved as charts in `reports/figures/`:
1. **Target Distribution Audit:** Quantifying the severe class imbalance.
2. **Transaction Activities:** Analyzing wallet interaction frequencies and smart contract call patterns.
3. **Balance Extremes:** Auditing heavily skewed token balances.
4. **Micro Risk-Cliffs Gapping:** Identifying specific threshold breaches in wallet health profiles.
5. **Macro Market Regime Correlations:** Evaluating systemic risk exposure against broader market movements.
6. **MI/KS Feature Shortlist:** Filtering the strongest predictive signals using Mutual Information and Kolmogorov-Smirnov statistics.

### Key Insights
Through our EDA, we uncovered that a micro feature like `max_risk_factor` exhibits a sharp inflection point cliff exactly at 0.7. Additionally, our macro indicators heavily suffer from Spearman multicollinearity, demanding defensive modeling strategies later in the pipeline.

## 3. DATA PREPARATION

### Data Cleaning
Implemented modularly in `src/preprocessing.py`, our cleaning pipeline utilizes dynamic capping and clamping mechanisms. This neutralizes extreme Whale wallet outliers without discarding valuable observations. We follow this with `log1p` distribution smoothing to normalize heavily skewed financial features and employ a strict `np.nan_to_num` defense mechanism to handle missing protocol data effectively.

### Feature Engineering & Feature Encoding
We engineered 52 highly robust features derived directly from on-chain behavior, encapsulating:
- **Gas Economics:** Historical gas consumption indicating urgency or willingness to pay block premiums.
- **Multi-Protocol Diversity Counts:** Interaction breadth across various DeFi dApps.
- **Net Cashflow Waterfalls:** Aggregated inflows versus outflows over specific time horizons.
- **Directional Transaction Velocity:** The speed and trajectory of capital movement from the wallet.

### Train-Test Split
To prevent data leakage and simulate true market stress testing, we justified a rigid Out-of-Time (OOT) chronological temporal split. The Training set consists of historical protocol data generated prior to 2023, while the Test set evaluates the model strictly on unseen 2023 Crypto Winter data.

## 4. MODEL DEVELOPMENT

### Candidate Models
We contrasted two distinct architectural paradigms tailored for non-linear risk surfaces:
- **Random Forest Classifier:** A Bagging approach focusing on variance reduction. Excellent for handling raw financial noise and complex feature interactions without extensive tuning.
- **XGBoost Classifier:** A Boosting approach focusing on bias reduction. Chosen for its mathematical superiority in optimizing custom loss functions and aggressively stepping down gradient errors on imbalanced datasets.

### Model Training
Inside `src/train.py`, we implemented defensive execution using the `Optuna` framework. We specifically optimized for **PR-AUC (Average Precision)** across a rigid 3-Fold Expanding Window Time-Series Cross-Validation. To combat the severe Spearman multicollinearity uncovered in our macro features, XGBoost was tuned with specialized L1/L2 regularization terms (`reg_alpha` / `reg_lambda`).

## 5. MODEL EVALUATION

### Model Performance Metrics

| Model Architecture |  ROC-AUC   |   PR-AUC   |  F1-Score  |   Recall   | Precision  |
| :----------------- | :--------: | :--------: | :--------: | :--------: | :--------: |
| **Random Forest**  |   0.9183   |   0.8490   |   0.7411   |   0.7948   |   0.6943   |
| **XGBoost**        | **0.9200** | **0.8520** | **0.7496** | **0.8064** | **0.7003** |

### Visual Evaluation
Our dual-panel diagnostic chart, available at `reports/figures/model_evaluation_curves.png`, plots the ROC Curve (Left) alongside the Precision-Recall Curve (Right). The PR Curve visually highlights XGBoost's premium lift and superior discriminative power specifically across the critical 60%–80% Recall horizon.

### Best Model Selection & Justification
XGBoost was formulated as the indisputable Champion Model. The core business justification rests on its **+1.16% lift in Recall** over Random Forest. In the context of decentralized credit, catching this additional fraction of bad debt translates directly to preventing multi-million dollar capital losses and mitigating protocol insolvency risks during sudden crashes.

## 6. MODEL INTERPRETATION

### Final Model Interpretation
To ensure the black-box model adheres to sound financial economic laws, we utilized a game-theoretic interpretation framework. Documented in `reports/figures/shap_summary_plot.png`, TreeExplainer isolates exact Shapley values. The global interpretation uncovers the highest contributing features driving default probability, heavily weighting `max_risk_factor` and historical debt/repayment discipline ratios. This mathematical validation confirms that the model is learning fundamentally sound risk profiles rather than spurious statistical noise.

## 7. BUSINESS RECOMMENDATIONS AND LIMITATIONS

### Actionable Recommendations
1. **Dynamic Interest Rate Curves:** Lending protocols should adopt dynamic borrowing rates intrinsically linked to wallet-level SHAP risk parameters, taxing high-risk behaviors algorithmically.
2. **Pre-emptive Liquidation Warnings:** Implement on-chain or off-chain oracle warnings triggered the exact moment a wallet approaches the statistically critical 0.7 micro-cliff threshold, incentivizing rapid self-repayment.

### Limitations
- **Single-Chain Bottleneck:** The current data pipeline and model are strictly bounded by Ethereum Mainnet history. It does not account for interconnected margin positions.
- **Hardware Boundary Constraints:** Substantial computation boundary constraints and severe hardware thermal throttling limited the breadth of our Optuna hyperparameter grid search.

### Future Work
- **Cross-Chain Risk Contagion:** Expanding the pipeline to track wallet contagion across rapidly scaling L2 rollups (e.g., Arbitrum, Optimism, Base).
- **Real-Time Streaming Pipelines:** Transitioning batch analytics into continuous, real-time risk assessment engines leveraging the massive throughput of Polars LazyFrames.
