import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main() -> None:

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    ans_raw = train_df["Transported"]

    train_features = train_df.drop(columns=['Transported'])
    test_features = test_df

    num_features = train_features.select_dtypes(include="number").columns.tolist()
    cat_features = train_features.select_dtypes(exclude="number").columns.tolist()

    ans_mapping = {"false": 0, "true": 1}
    if pd.api.types.is_bool_dtype(ans_raw):
        ans = ans_raw.astype(int)
    else:
        ans = ans_raw.astype(str).str.strip().str.lower().map(ans_mapping)


    num_transformer = Pipeline(
        steps=[ ("imputer", SimpleImputer(strategy="median")), ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("to_str", FunctionTransformer(lambda x: x.astype(str))),
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer,num_features ),
            ("cat", cat_transformer, cat_features),
        ]
    )

    model = RandomForestClassifier( n_estimators=300, random_state=42, n_jobs=-1)

    clf = Pipeline(
        steps=[
            ("process", preprocessor),
            ("model", model),
        ]
    )

    para_train, para_check, ans_train, ans_check = train_test_split(
        train_features,
        ans,
        test_size=0.2,
        random_state=42,
        stratify=ans, # stratified sampling
    )

    clf.fit(para_train, ans_train)

    check_pred = clf.predict(para_check)
    accuracy = accuracy_score( ans_check, check_pred )
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    test_pred = clf.predict(test_features).astype(bool)
    submission = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Transported": test_pred,
        }
    )
    submission.to_csv("submission.csv", index=False)

if __name__== "__main__":
    main()


