# Application Testing — Customer Churn Prediction

## 1. Purpose

The Streamlit application was tested using multiple customer profiles to verify that the deployed prediction workflow accepts different inputs and produces a probability, classification, and risk level without runtime errors.

## 2. Test Cases

| Test Case | Scenario | Churn Probability | Threshold | Prediction | Risk Level | Result |
|---|---|---:|---:|---|---|---|
| TC-01 | Standard customer profile | 26.35% | 33.80% | No Churn | Lower Risk | Pass |
| TC-02 | Higher-risk customer profile | 72.93% | 33.80% | Churn | Higher Risk | Pass |
| TC-03 | Zero-tenure edge case | 3.73% | 33.80% | No Churn | Lower Risk | Pass |

## 3. Test Case Details

### TC-01 — Standard Customer

The application generated a churn probability of 26.35%.

Since the probability was below the 33.80% decision threshold, the application classified the customer as No Churn.

**Result:** Pass.

### TC-02 — Higher-Risk Customer

The application generated a churn probability of 72.93%.

Since the probability exceeded the 33.80% decision threshold, the application classified the customer as Churn.

**Result:** Pass.

### TC-03 — Zero-Tenure Edge Case

A customer profile with zero months of tenure and zero total charges was entered.

The application successfully processed the input and generated a churn probability of 3.73%.

The application classified the customer as No Churn because the probability was below the 33.80% decision threshold.

**Result:** Pass.

## 4. Validation Checks

The following application behaviors were verified:

- Customer input fields accept valid categorical values.
- Numeric inputs are accepted correctly.
- Zero-tenure input is handled without errors.
- Zero total charges are handled without errors.
- The trained XGBoost model produces a probability.
- The optimized classification threshold is applied.
- The application displays the prediction.
- The application displays the risk level.
- The probability progress bar works correctly.
- No runtime error occurred during the completed tests.

## 5. Important Interpretation Note

The predictions produced by the application are model outputs based on patterns learned from the training data.

A predicted churn probability should not be interpreted as a guarantee that a customer will or will not churn.

The application is intended as a demonstration of a machine learning prediction workflow.