import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    root_mean_squared_log_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def main() -> None:

    # read data files
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    #get the result of training data
    ans = train_df["SalePrice"]

    # ignore low related feature columns
    train_features = train_df.drop(columns=["SalePrice", "Id"])
    test_features = test_df.drop(columns=["Id"])

    # divide data into num and cat
    num_features = train_features.select_dtypes(include=["number"]).columns.tolist()
    cat_features = train_features.select_dtypes(exclude=["number"]).columns.tolist()

    # feature dealing pipline
    num_transfer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_transfer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    #data pre processing
    data_process = ColumnTransformer(
        [
            ("num", num_transfer, num_features),
            ("cat", cat_transfer, cat_features),
        ]
    )

    #model setup
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rgs = Pipeline(
        steps=[
            ("process", data_process),
            ("model", model),
        ]
    )

    #divide training and validation data
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        train_features,
        ans,
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

    test_pred = rgs.predict(test_features)
    submission = pd.DataFrame(
        {
            "Id": test_df["Id"],
            "SalePrice": test_pred,
        }
    )
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
