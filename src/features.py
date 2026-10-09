"""
Feature Definitions and Preprocessing Pipeline for DetectX.

This module defines the feature schema, handles categorical encoding and scaling,
and documents feature rationale and known dataset limitations.
"""

from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Target and Excluded Columns
TARGET_COLUMN: str = "isFraud"
HEURISTIC_FLAG: str = "isFlaggedFraud"
IDENTIFIER_COLUMNS: List[str] = ["nameOrig", "nameDest"]

# Selected Feature Columns
CATEGORICAL_FEATURES: List[str] = ["type"]
NUMERIC_FEATURES: List[str] = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

# All features used for modeling
ALL_FEATURES: List[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Optimal low-memory data types for initial CSV reading
CSV_DTYPES = {
    "step": "uint16",
    "type": "category",
    "amount": "float32",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "uint8",
}

FEATURE_DOCUMENTATION = """
FEATURE CHOICES AND METHODOLOGICAL LIMITATIONS:
================================================
1. Included Features:
   - 'type': Categorical transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER).
     Encoded using OneHotEncoder(handle_unknown='ignore').
     LIMITATION: In the PaySim synthetic generator, fraud is strictly simulated inside
     TRANSFER and CASH_OUT operations. In real-world banking environments, fraud can occur
     across credit cards, point-of-sale payments, or ACH debits. Models trained here will
     learn a strong synthetic dependency on this transaction type.

   - 'amount': Transaction amount in currency. Scaled using StandardScaler.
     LIMITATION: Distribution is severely right-skewed. Real fraud may involve small micro-charges
     (card testing) or large lump sums.

   - 'step': Discrete simulation hour (1 to 743, spanning ~31 days). Scaled with StandardScaler.
     LIMITATION: Real financial transaction temporal patterns have complex periodic seasonality
     (time of day, day of week, payroll cycles) not fully represented by a simple linear step.

   - 'oldbalanceOrg', 'newbalanceOrig': Sender account balances. Scaled with StandardScaler.
     Signal: Fraudsters often drain the victim's account to zero.

   - 'oldbalanceDest', 'newbalanceDest': Recipient account balances. Scaled with StandardScaler.
     Signal: In fraud cases, recipient balances often show immediate cash out or 0 merchant balances.
     LIMITATION: Merchant accounts (starting with 'M') do not track balances (appear as 0.0).

2. Excluded Columns:
   - 'isFraud': Ground truth target variable. Excluded from feature space.
   - 'isFlaggedFraud': Baseline heuristic rule (transfer > 200,000). Excluded to prevent data leakage
     and avoid learning reliance on an ineffective existing rule.
   - 'nameOrig', 'nameDest': High-cardinality alphanumeric IDs. Excluded initially to avoid extreme
     cardinality explosion and prevent overfitting to simulated individual ID strings.
"""


def get_preprocessor() -> ColumnTransformer:
    """
    Creates and returns a scikit-learn ColumnTransformer.
    - Numerical features are standardized using StandardScaler.
    - Categorical features are one-hot encoded using OneHotEncoder.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return preprocessor
