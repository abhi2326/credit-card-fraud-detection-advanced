import pandas as pd
from src.features import split_features_target, build_preprocessor

def test_split_and_transform():
    df = pd.DataFrame({"Time": [1., 2., 3.], "Amount": [5., 6., 7.],
                       "V1": [0.1, 0.2, 0.3], "Class": [0, 1, 0]})
    X, y = split_features_target(df)
    transformed = build_preprocessor(X).fit_transform(X)
    assert len(y) == 3
    assert transformed.shape[0] == 3
