import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def age_group(age):
	if age <12:
		return "Child"
	elif age <18:
		return "Teenager"
	elif age <35:
		return "Young"
	elif age <60:
		return "Adult"
	else:
		return "Senior"
	
def family_size(sib, par):
	return sib + par + 1

def name_spe(name):
	if "Mrs." in name:
		return "Mrs."
	elif "Master." in name:
		return "Master."
	elif "Miss." in name:
		return "Miss."
	elif "Mr." in name:
		return "Mr."
	elif "Col." in name:
		return "Col."
	elif "Dr." in name:
		return "Dr."
	elif "Rev." in name:
		return "Rev."
	elif "Major." in name:
		return "Major."
	elif "Mlle." in name:
		return "Mlle."
	elif "Mme." in name:
		return "Mrs."
	elif "Sir." in name:
		return "Sir."
	

def main() -> None:
	train_df = pd.read_csv("train.csv")
	test_df = pd.read_csv("test.csv")

	y = train_df["Survived"]

	# ignore name
	train_features = train_df.drop(columns=["Survived"])
	test_features = test_df

	# mapping sex to 0 and 1
	sex_mapping = {"male": 0, "female": 1}
	train_features["Sex"] = train_features["Sex"].map(sex_mapping)
	test_features["Sex"] = test_features["Sex"].map(sex_mapping)

	# create age group column， doing age feature
	train_features["AgeGroup"] = train_features["Age"].apply(age_group)
	test_features["AgeGroup"] = test_features["Age"].apply(age_group)
    # drop the original age column
	train_features = train_features.drop(columns=["Age"])
	test_features = test_features.drop(columns=["Age"])
	
    # create family size column, doing sibsp and parch feature
	train_features["FamilySize"] = train_features.apply(lambda row: family_size(row["SibSp"], row["Parch"]), axis=1)
	test_features["FamilySize"] = test_features.apply(lambda row: family_size(row["SibSp"], row["Parch"]), axis=1)

    #identify spetial name
	train_features["NameSpe"] = train_df["Name"].apply(name_spe)
	test_features["NameSpe"] = test_df["Name"].apply(name_spe)
	#drop name column
	train_features = train_features.drop(columns=["Name"])
	test_features = test_features.drop(columns=["Name"])


	# identify numeric and categorical features
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
	

	# Refit with full training data before generating test predictions.
	clf.fit(train_features, y)

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
