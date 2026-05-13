from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    root_mean_squared_log_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "store-sales-time-series-forecasting"

def add_features(data: pd.DataFrame, store: pd.DataFrame, oil: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    oil = oil.copy()
    oil["date"] = pd.to_datetime(oil["date"])

    data = data.merge(store, on="store_nbr", how="left")
    data = data.merge(oil, on="date", how="left")

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["dayofweek"] = data["date"].dt.dayofweek
    return data

def main() -> None:

    #read the info files
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    #special info
    store = pd.read_csv(DATA_DIR / "stores.csv")
    oil = pd.read_csv(DATA_DIR / "oil.csv")

    train = add_features(train, store, oil)
    test = add_features(test, store, oil)

    #learning result
    result = train["sales"]
    train = train.drop(columns=["id", "date", "sales"])
    test_ids = test["id"]
    test = test.drop(columns=["id", "date"])

    #features: item_category, date, earthquake check, oil price, store category, promotion
    #info dividing
    num_info = train.select_dtypes(include=["number"]).columns.tolist()
    cat_info = train.select_dtypes(exclude=["number"]).columns.tolist()

    #feature dealing
    num_transfer = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_transfer = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    #data pre-dealing
    data_process = ColumnTransformer(
        [
            ("num", num_transfer, num_info),
            ("cat", cat_transfer, cat_info),
        ]
    )

    model = HistGradientBoostingRegressor(random_state=42)
    rgs = Pipeline(
        steps=[
            ("process", data_process),
            ("model", model),
        ]
    )

    X_train, X_valid, Y_train, Y_valid = train_test_split(
        train,
        result,
        test_size=0.15,
        random_state=42,
    )

    #begin training
    rgs.fit(X_train, Y_train)

    # check validation metrics
    valid_pred = rgs.predict(X_valid)

    rmse = root_mean_squared_error(Y_valid, valid_pred)
    # RMSLE requires non-negative values.
    valid_pred_clipped = np.clip(valid_pred, a_min=0, a_max=None)
    rmsle = root_mean_squared_log_error(Y_valid, valid_pred_clipped)
    mae = mean_absolute_error(Y_valid, valid_pred)

    print(f"Validation MAE: {mae:.2f}")
    print(f"Validation RMSE: {rmse:.2f}")
    print(f"Validation RMSLE: {rmsle:.5f}")

    test_pred = rgs.predict(test)
    submission = pd.DataFrame(
        {
            "id": test_ids,
            "sales": np.clip(test_pred, a_min=0, a_max=None),
        }
    )
    submission.to_csv(BASE_DIR / "submission.csv", index=False)

if __name__ == "__main__":
    main()
