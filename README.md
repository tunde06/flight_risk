# Flight Risk Score Model
## Executive Notebook Narration and End-to-End Machine Learning Workflow

### Purpose

This document provides an executive and technical narration of the **Flight Risk Score Model**, an end-to-end machine learning solution designed to identify employees who may be at elevated risk of **voluntary termination within the next 90 days**.

The workflow was developed as part of an AI Center of Excellence (AI CoE) data science mentorship project. It documents the business rationale, analytical methodology, model-development lifecycle, explainability approach, governance controls, and deployment path used to transform workforce data into an actionable Flight Risk Score.

The solution is designed as a **decision-support tool**. Predictions should be interpreted alongside HRBP, manager, and business context and should not be used as the sole basis for employment decisions.

---

## Executive Summary

The Flight Risk Score Model addresses a common workforce challenge: voluntary turnover is often recognized only after an employee has decided to leave. The model shifts the process from retrospective reporting toward **proactive retention intelligence** by estimating the probability that an active employee may voluntarily terminate within the next three months.

The workflow evaluates multiple classification approaches—including Logistic Regression, Random Forest, XGBoost, soft-voting ensembles, and stacking—to identify the model that best separates potential leavers from stayers. Performance is assessed using metrics appropriate for an imbalanced workforce outcome, including **ROC-AUC, recall, precision, F1 score, confusion matrices, and Precision-Recall analysis**.

The strongest individual model achieved approximately **0.876 ROC-AUC** in the development run documented here. Model outputs are converted into continuous risk probabilities that can be expressed as a **0–100 Flight Risk Score** and grouped into Low, Medium, and High risk tiers for prioritization.

The model is supported by explainability techniques, including feature importance and SHAP, so HR users can understand the factors contributing to a prediction. The final model artifact is versioned through MLflow and Unity Catalog, supporting reproducibility, auditability, batch scoring, and future serving workflows.

> **Important development note:** The current notebook uses synthetic employee data designed to approximate realistic organizational patterns. Model performance on synthetic data should not be interpreted as expected production performance. Production deployment requires validation, fairness review, temporal testing, monitoring, and governance using approved organizational data.

---

# End-to-End ML Lifecycle

```text
Business Problem Definition
        |
        v
Synthetic Workforce Data Generation
        |
        v
Data Quality & Preprocessing
        |
        v
Exploratory / Descriptive Analysis
        |
        v
Feature Engineering & Target Definition
        |
        v
Train / Validation / Test Strategy
        |
        v
Class-Imbalance Treatment
        |
        v
Candidate Model Training
        |
        v
Model Evaluation & Comparison
        |
        v
Best-Model Selection
        |
        v
Explainability (Feature Importance + SHAP)
        |
        v
Flight Risk Score & HRBP Decision Support
        |
        v
Model Persistence & MLflow / Unity Catalog Registry
        |
        v
Batch Scoring / Serving Endpoint
        |
        v
Monitoring, Validation & Responsible-AI Review
```

---

# Phase 1 — Environment Setup and Reproducibility

**Objective:** Establish a controlled analytical environment in which dependencies, random processes, and model outputs can be reproduced consistently.

| Step | Notebook Activity | Technical Purpose | Business / Governance Significance |
| --- | --- | --- | --- |
| 1 | Install Dependencies | Installs required Python packages such as pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, MLflow, matplotlib, and related utilities. | Creates a consistent analytical toolchain and reduces environmental differences across development and reruns. |
| 2 | Import Libraries | Loads packages for data preparation, visualization, modeling, evaluation, explainability, tracking, and deployment. | Documents the analytical stack used to build and operationalize the model. |
| 3 | Set Random Seed | Sets a fixed seed such as `np.random.seed(42)` for stochastic data generation and selected modeling processes. | Improves reproducibility during development and supports controlled comparison across experiments. It does not, by itself, guarantee full model reproducibility. |

### Executive Interpretation

This phase establishes the foundation for a repeatable model-development process. Reproducibility is important for model governance because analysts must be able to explain how a result was produced, recreate an experiment, and distinguish true model improvements from random variation.

---

# Phase 2 — Workforce Data Generation and Data Foundation

**Objective:** Build a privacy-safe development dataset that mimics important workforce relationships and provides a controlled environment for model prototyping.

| Step | Notebook Activity | Technical Purpose | Business / Governance Significance |
| --- | --- | --- | --- |
| 4 | Generate Synthetic Employee Data | Creates approximately 10,000 employee records with workforce attributes such as tenure, management level, job family, performance, compensation, promotion, manager score, growth, workload, engagement, and satisfaction. | Enables model development without exposing real employee records while preserving plausible workforce relationships for demonstration and testing. |
| 5 | Define Future Resignation Target | Generates `resign_3m`, representing whether an employee voluntarily terminates within the modeled 90-day prediction window. | Establishes the business outcome the model is intended to predict. |
| 6 | Save Raw Dataset | Persists the development dataset to governed storage such as a Unity Catalog Volume. | Creates a traceable source artifact for downstream preprocessing and model development. |
| 7 | Document Alternative Generation Logic | Retains prior or alternate synthetic-data approaches as development history where appropriate. | Supports transparency around iterative design decisions; production notebooks should archive rather than clutter the main workflow with obsolete code. |

### Target Definition

The recommended model population is:

- employees who are **active at the snapshot date**, and
- whose future voluntary-termination outcome can be observed over the next 90 days.

The target is:

```python
resign_3m
```

where:

- `0` = employee remained during the prediction window
- `1` = employee voluntarily terminated within the next 3 months

`active_status` should generally be used to define the eligible population rather than as the prediction target.

### Executive Interpretation

The model is intended to answer a forward-looking question:

> **Among employees who are active today, who appears most likely to voluntarily leave within the next 90 days?**

This distinction is important because predicting whether someone has already left is a retrospective classification problem—not a flight-risk prediction problem.

---

# Phase 3 — Data Preparation, Feature Governance, and Encoding

**Objective:** Convert the raw employee dataset into a model-ready analytical table while excluding identifiers or fields that could introduce unnecessary noise, leakage, or governance concerns.

| Step | Notebook Activity | Technical Purpose | Business / Governance Significance |
| --- | --- | --- | --- |
| 8 | Remove Non-Predictive Identifiers | Excludes `employee_id` and raw snapshot/date identifiers from model features. | Prevents the model from learning meaningless employee-specific patterns. IDs can be retained separately for mapping predictions back to records. |
| 9 | Review Organizational Fields | Assesses business unit, location, job family, and management level before modeling. | Helps determine whether organizational variables add legitimate predictive value or introduce unnecessary high-cardinality complexity. |
| 10 | Encode Categorical Features | Converts categorical fields into model-consumable representations. | Allows models to use job and organizational information consistently. One-hot encoding or native categorical handling is generally preferable to arbitrary label encoding for nominal categories. |
| 11 | Save Model-Ready Dataset | Persists the encoded or transformed dataset for repeatable downstream training. | Creates a reusable intermediate artifact and supports retraining pipelines. |

### Important Encoding Note

`management_level` may contain a genuine hierarchy, but `functional_job_family` is typically nominal. Assigning integer labels to nominal categories can introduce an artificial ordering. For production modeling, use one of the following where appropriate:

- one-hot encoding,
- target-safe categorical encoding performed inside cross-validation, or
- native categorical support in algorithms such as LightGBM/CatBoost.

---

# Phase 4 — Data Quality and Exploratory Analysis

**Objective:** Validate the analytical dataset, understand the workforce population, quantify class imbalance, and identify relationships that may influence voluntary turnover.

## 4.1 Data Quality Checks

| Analysis | Purpose | Executive Relevance |
| --- | --- | --- |
| Dataset Shape | Confirms expected rows and columns. | Verifies preprocessing did not unexpectedly remove records. |
| Missing Values | Measures missingness by field. | Identifies variables that may require imputation or may be unreliable for decision support. |
| Data Types | Confirms model-compatible feature types. | Prevents avoidable pipeline failures and surfaces inconsistent source fields. |
| Duplicate Records | Tests for duplicate employee/snapshot combinations. | Helps prevent duplicated observations from biasing training results. |
| Range / Validity Checks | Tests values such as compensation, tenure, scores, and target values. | Supports data-quality governance and protects against invalid inputs. |

## 4.2 Recommended Descriptive Analysis

### Workforce KPI Cards

Present a small set of headline indicators before detailed charts:

| KPI | Example |
| --- | ---: |
| Employee Population | 10,000 |
| Active Workforce | 9,200 |
| Voluntary Terminations / Positive Target | 1,532 |
| Stayers / Negative Target | 8,468 |
| Voluntary Turnover Rate | 15.3% |

> The exact KPI denominator should match the modeling population. If the model is trained only on active employees at snapshot, calculate the target rate using that eligible population.

### Target Distribution

**Best presentation:** count and percentage bar chart.

Purpose: make the class imbalance immediately visible and show why accuracy alone is insufficient.

### Age Distribution by Target

**Best presentation:** histogram or density plot by `resign_3m`, plus a box plot for side-by-side comparison.

Purpose: determine whether voluntary leavers cluster within particular age bands. Age should be handled carefully in production because of fairness and employment-law considerations.

### Tenure Distribution by Target

**Best presentation:** histogram/density plot and turnover-rate bar chart by tenure band.

Purpose: identify early-tenure, mid-tenure, or long-tenure risk concentrations.

### Turnover by Management Level

**Best presentation:** ranked horizontal bar chart showing turnover **rate**, not only counts.

Purpose: distinguishes workforce size effects from true elevated risk.

### Turnover by Job Family / Business Unit

**Best presentation:** top-N ranked rate chart with minimum-population thresholds.

Purpose: identifies organizational segments with disproportionately high turnover while avoiding misleading rates from very small groups.

### Engagement, Satisfaction, Manager, Growth, and Workload Scores

**Best presentation:** box plots or grouped means by target, supplemented by turnover rates across score bands.

Purpose: connects employee-experience indicators with observed resignation behavior.

### Promotion and Compensation Indicators

**Best presentation:** grouped turnover-rate charts for promotion status, compensation range, and pay-change bands.

Purpose: explores whether growth and reward patterns are associated with future turnover.

### Correlation / Association Analysis

**Best presentation:** correlation heatmap for numeric features plus targeted statistical tests where needed.

Purpose: identifies redundancy and helps analysts understand relationships among predictors. Correlation should not be interpreted as causation.

### Recommended Executive Summary Table

A useful leadership view is a side-by-side profile of stayers and leavers:

| Indicator | Stayers | Leavers | Interpretation |
| --- | ---: | ---: | --- |
| Average Tenure | Example | Example | Highlights tenure concentration |
| Engagement | Example | Example | Shows employee-experience difference |
| Growth Score | Example | Example | Indicates development opportunity gap |
| Promotion Rate | Example | Example | Highlights career-mobility differences |
| Manager Score | Example | Example | Indicates manager-experience differences |

---

# Phase 5 — Feature Engineering and Target Preparation

**Objective:** Create analytically meaningful predictors while avoiding leakage from information that would not have been known at the scoring date.

Recommended feature families include:

- tenure and time-in-role indicators,
- promotion and internal-mobility history,
- performance and performance trend,
- compensation position and recent pay change,
- engagement and satisfaction,
- manager effectiveness,
- growth / career opportunity indicators,
- workload or burnout indicators,
- job family and selected organizational context.

### Leakage Control

Every feature should answer this question:

> **Would this information have been available on or before the employee's snapshot/scoring date?**

Features created using post-snapshot events must be excluded because they leak future information into the model.

---

# Phase 6 — Train/Test Strategy and Class-Imbalance Treatment

**Objective:** Estimate true out-of-sample performance and ensure the minority resignation class receives sufficient modeling attention.

## Train/Test Split

A typical development split is:

```text
Training Set: 80%
Holdout Test Set: 20%
```

Use stratification so the resignation rate is represented consistently across train and test sets.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

For a production flight-risk model with repeated employee snapshots, a **temporal holdout** is stronger than a random split because it better simulates predicting future turnover from past data.

## Class Imbalance

The target may contain substantially fewer leavers than stayers. Techniques that can be evaluated include:

- class weighting,
- SMOTE applied **only to the training folds**,
- threshold tuning,
- precision-recall optimization,
- cost-sensitive modeling.

### Important Governance Note on SMOTE

SMOTE should never be applied before the train/test split. When cross-validation is used, oversampling should occur inside an imbalanced-learn pipeline so synthetic observations are generated only from each training fold. This prevents validation leakage.

---

# Phase 7 — Candidate Model Development

**Objective:** Compare multiple algorithms rather than assuming a single modeling technique will be best.

## Logistic Regression — Interpretable Baseline

**Role:** Establish a transparent benchmark against which more complex models can be measured.

**Strengths:**

- easy to interpret,
- efficient,
- useful for coefficient-based directionality,
- strong baseline for binary outcomes.

## Random Forest — Nonlinear Ensemble

**Role:** Capture nonlinear relationships and interactions using an ensemble of decision trees.

**Strengths:**

- handles complex interactions,
- robust to many feature distributions,
- provides feature importance.

## XGBoost — Gradient-Boosted Trees

**Role:** Provide a high-performing nonlinear classifier that sequentially corrects prior model errors.

**Strengths:**

- strong predictive performance on structured/tabular data,
- captures interactions automatically,
- offers regularization and flexible class handling,
- integrates well with SHAP explainability.

## Ensemble Models

### Soft Voting

Combines predicted probabilities from multiple models. This can improve stability when component models make different types of errors.

### Stacking

Uses a meta-model to learn how to combine predictions from base classifiers. Stacking can improve performance, but additional complexity should be justified by meaningful validation gains.

---

# Phase 8 — Standardized Model Evaluation

**Objective:** Compare all candidate models using consistent metrics that reflect both discrimination and the operational cost of false positives and false negatives.

## Core Metrics

| Metric | What It Measures | Why It Matters for Flight Risk |
| --- | --- | --- |
| ROC-AUC | Ability to rank leavers above stayers across thresholds. | Measures overall discrimination independent of one fixed cutoff. |
| Recall / Sensitivity | Share of actual leavers correctly identified. | Indicates how many true resignation cases the model catches. |
| Precision | Share of flagged employees who actually leave. | Helps manage unnecessary HR interventions and false alarms. |
| F1 Score | Balance of precision and recall. | Useful when both missed leavers and false alerts matter. |
| Average Precision / PR-AUC | Precision-recall performance across thresholds. | Particularly informative when resigners are a minority class. |
| Accuracy | Overall share correctly classified. | Useful context, but potentially misleading with class imbalance. |
| Confusion Matrix | Counts true/false positives and negatives. | Translates model performance into operational volumes. |

## Recommended Evaluation Visuals

1. ROC Curve
2. Precision-Recall Curve
3. Confusion Matrix
4. Threshold Performance Table
5. Lift / Gains Chart
6. Calibration Plot

### Documented Development Results

The model-development run summarized in this notebook produced approximately:

| Model | ROC-AUC |
| --- | ---: |
| Logistic Regression | 0.826 |
| Random Forest | 0.862 |
| XGBoost | **0.876** |

These results should be treated as development evidence on synthetic data rather than production performance estimates.

### Interpreting ROC-AUC Correctly

An AUC of 0.876 does **not** mean the model is 87.6% accurate. It means that, when randomly comparing one positive case with one negative case, the model will assign the positive case a higher risk score approximately 87.6% of the time.

---

# Phase 9 — Best-Model Selection and Threshold Strategy

**Objective:** Select the model that best balances predictive performance, interpretability, operational usefulness, and governance requirements.

The best model should not be selected on ROC-AUC alone. Recommended selection criteria include:

- ROC-AUC,
- PR-AUC / Average Precision,
- positive-class recall,
- precision,
- calibration,
- stability across folds / time periods,
- subgroup performance,
- explainability,
- inference complexity,
- business capacity for intervention.

## Classification Threshold

The default probability threshold of 0.50 is not automatically optimal for retention use cases.

The final threshold should reflect the organization's cost tradeoff between:

- **false negatives:** missing employees who later resign, and
- **false positives:** initiating unnecessary retention interventions.

A threshold analysis should show precision, recall, F1, number of employees flagged, and estimated intervention volume across candidate cutoffs.

---

# Phase 10 — Model Explainability and Top-Driver Analysis

**Objective:** Explain both overall model behavior and individual employee predictions.

## Global Feature Importance

Ranks features according to their contribution to overall model behavior.

**Best presentation:** horizontal bar chart of the top 10–15 variables.

## SHAP Global Explanation

A SHAP summary / beeswarm plot shows:

- which features matter most,
- whether high or low values tend to raise or lower predicted risk,
- how feature effects vary across employees.

## SHAP Individual Explanation

For a selected employee, a waterfall or similar local explanation shows the variables pushing the prediction:

- toward higher flight risk, and
- toward lower flight risk.

### Executive Interpretation

Explainability converts the output from:

> “This employee has a 78% predicted risk.”

into:

> “The elevated score is primarily associated with lower growth opportunity, high workload, limited recent pay movement, and lack of recent promotion, partially offset by strong performance.”

This enables more informed HRBP review while maintaining human judgment.

---

# Phase 11 — Flight Risk Score and HRBP Decision Support

**Objective:** Translate statistical probability into a practical, governed decision-support experience.

A model probability can be expressed as:

```python
flight_risk_score = predicted_probability * 100
```

Example risk tiers:

| Flight Risk Score | Risk Tier | Suggested Response |
| --- | --- | --- |
| 0–39 | Low | Standard talent-management practices and normal monitoring |
| 40–69 | Medium | Proactive manager/HRBP review and development discussion |
| 70–100 | High | Prioritized contextual review and targeted retention assessment |

## HRBP Recommendation Framework

The Retention Agent can organize decision support into nine sections:

1. Employee Risk Summary
2. Key Drivers Behind the Prediction
3. Contextual Interpretation for the HRBP
4. Comparison to Relevant Peer Group
5. Data Quality and Confidence Notes
6. Suggested HRBP Assessment Questions
7. Recommended Follow-Up Actions
8. Suggested Conversation Guidance
9. Risk Interpretation Summary

### Responsible-Use Principle

The model should **surface risk for review—not prescribe employment action**. HRBPs and managers should validate context before taking action, and sensitive/protected characteristics should be governed appropriately.

---

# Phase 12 — Model Persistence and Artifact Validation

**Objective:** Save the selected model and supporting metadata so the exact trained artifact can be reused for scoring and deployment.

Recommended artifacts include:

- serialized model (`.joblib` / `.pkl`),
- model feature list,
- preprocessing pipeline,
- threshold configuration,
- evaluation metrics,
- feature-importance outputs,
- training metadata,
- package/environment dependencies.

After persistence, reload the saved model and run a small scoring test to verify artifact integrity.

---

# Phase 13 — MLflow Tracking and Unity Catalog Model Registry

**Objective:** Convert the trained model from a notebook object into a traceable enterprise ML asset.

Recommended MLflow / Unity Catalog steps include:

1. Log parameters and model metrics.
2. Log the preprocessing/model pipeline.
3. Store input examples and model signatures.
4. Register the model under a governed Unity Catalog namespace.
5. Record the model version and training run.
6. Associate validation artifacts with the registered version.

Example registry path:

```text
lab_teams.pa.flightScore_model_v2
```

### Business / Governance Significance

Registration provides:

- lineage,
- model versioning,
- controlled promotion,
- rollback capability,
- discoverability,
- reproducible deployment.

Pinned package versions can improve reproducibility, but dependencies should be managed through an approved environment strategy rather than treated as permanent production requirements without security and compatibility review.

---

# Phase 14 — Batch Scoring and Real-Time Serving

**Objective:** Make the model accessible to downstream workflows and applications.

## Batch Scoring

For workforce planning, periodic batch scoring is often the simplest and most operationally appropriate pattern—for example, generating refreshed employee risk scores monthly or after an approved HR data refresh.

## Real-Time Serving

A model-serving endpoint can expose predictions through an API for applications such as an HRBP Retention Agent.

The documented development run encountered an HTTP 403 permission issue while checking a serving endpoint. This is an infrastructure/access issue rather than a model-quality issue. Resolve by validating:

- endpoint name,
- model/version association,
- workspace and serving permissions,
- `CAN_QUERY` or equivalent access,
- service-principal/user authorization.

Do not characterize the model as fully production-ready until the intended scoring path has passed operational testing and governance approval.

---

# Phase 15 — Production Validation, Monitoring, and Responsible AI

**Objective:** Ensure model performance remains reliable, fair, explainable, and operationally useful after deployment.

Production controls should include:

## Data Monitoring

- missingness and schema changes,
- feature distribution drift,
- category changes,
- scoring population changes.

## Model Monitoring

- ROC-AUC / PR-AUC over time,
- recall and precision,
- calibration,
- score distribution,
- threshold intervention volume,
- outcome drift.

## Fairness Review

Evaluate model performance across appropriately governed workforce groups before use. Protected or sensitive characteristics may be needed for **fairness auditing** even when they are intentionally excluded as model predictors.

## Human Oversight

Model outputs should be reviewed within existing talent-management and HR governance processes. High-risk classifications should not automatically trigger adverse employment decisions.

## Retraining Strategy

Define criteria for retraining based on:

- data drift,
- performance degradation,
- business/process changes,
- material changes in workforce composition,
- scheduled model-review cadence.

---

# Executive Model Scorecard

| Area | Development Status | Executive Interpretation |
| --- | --- | --- |
| Business Objective | Defined | Predict voluntary termination within a 90-day horizon |
| Development Dataset | Synthetic, ~10,000 records | Privacy-safe proof-of-concept environment |
| Target | `resign_3m` | Forward-looking voluntary termination indicator |
| Candidate Models | Logistic Regression, Random Forest, XGBoost, ensembles | Multiple approaches compared rather than assuming one algorithm |
| Best Documented ROC-AUC | ~0.876 | Strong discrimination in the synthetic development dataset |
| Explainability | Feature importance + SHAP | Supports global and employee-level interpretation |
| Risk Output | Probability / 0–100 score | Enables threshold-based prioritization |
| Model Registry | MLflow + Unity Catalog | Supports versioning and lineage |
| Serving | Endpoint workflow defined | Access/permission issue must be resolved and tested |
| Responsible AI | Human review required | Prediction is decision support, not an automated employment decision |
| Production Readiness | Proof-of-concept / pre-production | Requires validation on approved real data, fairness testing, monitoring, and operational approval |

---

# Leadership Takeaways

1. **The solution predicts a future business outcome.** The model estimates which currently active employees may voluntarily terminate within the next 90 days.
2. **The workflow addresses class imbalance and evaluates multiple algorithms.** This reduces reliance on misleading accuracy metrics and supports evidence-based model selection.
3. **The model produces continuous risk probabilities rather than only yes/no predictions.** This allows HR to align intervention thresholds with operational capacity and business priorities.
4. **Explainability is built into the solution.** Feature importance and SHAP help users understand both overall model behavior and individual predictions.
5. **The model is designed to support—not replace—HR judgment.** Employee-level predictions require contextual validation before action.
6. **The model has an enterprise MLOps path.** Artifacts can be persisted, registered through MLflow/Unity Catalog, batch scored, and ultimately served through an approved endpoint.
7. **Production deployment requires additional validation.** Synthetic-data results demonstrate the workflow, but production use requires temporal validation, fairness analysis, monitoring, governance approval, and testing on authorized organizational data.

---

# Recommended Repository Structure

```text
flight-risk-score-model/
|
|-- README.md
|-- notebooks/
|   |-- 01_environment_setup.py
|   |-- 02_data_preparation.py
|   |-- 03_exploratory_analysis.py
|   |-- 04_model_training.py
|   |-- 05_model_evaluation.py
|   |-- 06_explainability.py
|   |-- 07_model_registry_deployment.py
|
|-- src/
|   |-- data.py
|   |-- features.py
|   |-- train.py
|   |-- evaluate.py
|   |-- explain.py
|   |-- predict.py
|
|-- artifacts/
|   |-- model_comparison_metrics.csv
|   |-- threshold_analysis.csv
|   |-- feature_importance.csv
|
|-- tests/
|   |-- test_data.py
|   |-- test_prediction.py
|
|-- requirements.txt
|-- .gitignore
|-- LICENSE
```

> Do not commit sensitive employee data, credentials, model-serving tokens, or restricted model artifacts to a public or unauthorized GitHub repository.

---

# Suggested Notebook Section Order

For a single Databricks notebook, use this order:

1. Business Problem & Model Objective
2. Environment Setup
3. Data Generation / Data Load
4. Data Quality & Preprocessing
5. Exploratory Analysis
6. Feature Engineering
7. Target Definition
8. Train/Test Strategy
9. Class-Imbalance Treatment
10. Candidate Model Definition
11. Standardized Model Evaluation
12. Ensemble Modeling
13. Best-Model Selection
14. Threshold Analysis
15. Feature Importance
16. SHAP Explainability
17. Flight Risk Scoring
18. HRBP Recommendation / Retention Agent
19. Model Persistence
20. MLflow & Unity Catalog Registration
21. Batch / Endpoint Scoring
22. Monitoring, Governance & Next Steps

---

## Final Positioning

The Flight Risk Score Model should be positioned as a **governed predictive decision-support capability** that combines workforce analytics, machine learning, explainable AI, and MLOps. Its value is not simply in predicting who may leave, but in helping HR teams prioritize where additional context, development conversations, and retention review may be most valuable.
