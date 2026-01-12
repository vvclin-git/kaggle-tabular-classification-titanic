from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def create_preprocess_baseline() -> ColumnTransformer:
    num_cols = ["Age", "Fare", "SibSp", "Parch"]
    cat_cols = ["Sex", "Pclass", "Embarked"]
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", _make_ohe()),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocess


def create_preprocess_improved() -> ColumnTransformer:
    num_cols = ["Age", "Fare", "SibSp", "Parch", "Company"]
    cat_cols = ["Sex", "Pclass", "Embarked", "Title", "Deck"]
    bol_cols = ["HasCabin", "Alone"]
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", _make_ohe()),
                    ]
                ),
                cat_cols,
            ),
            ("bol", "passthrough", bol_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocess


def create_preprocess_xgb() -> ColumnTransformer:
    num_cols = ["Age", "Fare", "SibSp", "Parch", "Company"]
    cat_cols = ["Sex", "Pclass", "Embarked", "Title"]
    bol_cols = ["HasCabin", "Alone"]
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", _make_ohe()),
                    ]
                ),
                cat_cols,
            ),
            ("bol", "passthrough", bol_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocess
