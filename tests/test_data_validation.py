import pandas as pd
import pytest
from src.data_validation import validate_transactions

def test_valid_data_passes():
    df = pd.DataFrame({"Time": [1, 2], "Amount": [10.0, 20.0], "V1": [0.1, 0.2], "Class": [0, 1]})
    validate_transactions(df)

def test_negative_amount_fails():
    df = pd.DataFrame({"Time": [1], "Amount": [-1.0], "Class": [0]})
    with pytest.raises(ValueError):
        validate_transactions(df)

def test_non_binary_target_fails():
    df = pd.DataFrame({"Time": [1], "Amount": [10.0], "Class": [2]})
    with pytest.raises(ValueError):
        validate_transactions(df)
