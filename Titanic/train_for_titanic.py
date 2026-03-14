import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def main() -> None:
	train_df = pd.read_csv("train.csv")
	test_df = pd.read_csv("test.csv")

	y = train_df["Survived"]

	# 忽略 Name，其余列均作为特征。
	train_features = train_df.drop(columns=["Survived", "Name"])
	test_features = test_df.drop(columns=["Name"])

	# 统一把 Sex 转为 0/1。
	sex_mapping = {"male": 0, "female": 1}
	train_features["Sex"] = train_features["Sex"].map(sex_mapping)
	test_features["Sex"] = test_features["Sex"].map(sex_mapping)

	# 数值列和类别列分开处理：
	# - Age/Fare 等数值缺失值用中位数填充
	# - 类别缺失值用常量填充并做 one-hot
	numeric_features = train_features.select_dtypes(include=["number"]).columns.tolist()
	categorical_features = train_features.select_dtypes(exclude=["number"]).columns.tolist()

	numeric_transformer = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
		]
	)

	categorical_transformer = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
			("onehot", OneHotEncoder(handle_unknown="ignore")),
		]
	)

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_transformer, numeric_features),
			("cat", categorical_transformer, categorical_features),
		]
	)

	model = RandomForestClassifier(
		n_estimators=300,
		random_state=42,
		n_jobs=-1,
	)

	clf = Pipeline(
		steps=[
			("preprocessor", preprocessor),
			("model", model),
		]
	)

	X_train, X_valid, y_train, y_valid = train_test_split(
		train_features,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)

	clf.fit(X_train, y_train)

	valid_pred = clf.predict(X_valid)
	accuracy = accuracy_score(y_valid, valid_pred)
	print(f"Validation Accuracy: {accuracy * 100:.2f}%")

	test_pred = clf.predict(test_features)
	submission = pd.DataFrame(
		{
			"PassengerId": test_df["PassengerId"],
			"Survived": test_pred.astype(int),
		}
	)
	submission.to_csv("submission.csv", index=False)
	print("Saved submission file: submission.csv")


if __name__ == "__main__":
	main()
