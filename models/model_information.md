# Model Information

## Final Model

The final customer churn prediction model is an XGBoost classification model implemented as a scikit-learn pipeline.

The pipeline combines:

1. Numerical feature preprocessing using `StandardScaler`
2. Categorical feature encoding using `OneHotEncoder`
3. XGBoost classification

The complete pipeline is saved as:

`xgboost_churn_pipeline.pkl`

## Model Selection

Four classification algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The models were compared using cross-validation and evaluated on a separate held-out test set.

XGBoost achieved the highest cross-validation ROC-AUC among the evaluated models after hyperparameter tuning.

## Tuned XGBoost Configuration

The selected XGBoost model was obtained through GridSearchCV using ROC-AUC as the optimization metric.

The selected parameters include:

- Number of estimators: 100
- Learning rate: 0.05
- Maximum depth: 3
- Column subsampling: 0.8

The complete fitted pipeline is stored in the model artifact.

## Classification Threshold

The application uses an optimized probability threshold rather than the default 0.50 cutoff.

Saved threshold:

`0.33801376198160596`

Approximately:

`33.80%`

The threshold was selected using F1-score optimization with out-of-fold training probabilities.

## Prediction Logic

For a customer, the model produces a churn probability between 0 and 1.

The application then compares the probability with the saved threshold.

```text
Probability >= 0.33801376198160596
        ↓
      Churn

Probability < 0.33801376198160596
        ↓
     No Churn