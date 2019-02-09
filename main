import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_file_path = 'C:\\Users\\aibow\\Python Projects\\Titanic\\train (1).csv'
train_data = pd.read_csv(train_file_path)
test_file_path = 'C:\\Users\\aibow\\Python Projects\\Titanic\\test (1).csv'
test_data = pd.read_csv(test_file_path)

average_age = train_data["Age"].mean()

train_data = train_data.fillna(value={"Age":average_age})
test_data = test_data.fillna(value={"Age":average_age, "Fare":35})
print(test_data["Fare"].describe())
print(test_data["Pclass"].describe())
print(test_data["Sex"].describe())

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

for column in ["Pclass","Sex"]:
    train_data = create_dummies(train_data,column)
    test_data = create_dummies(test_data,column)

columns = ['Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'Age']
train_X, test_X, train_y, test_y = train_test_split(train_data[columns], train_data['Survived'], test_size=0.2, random_state=0)

# FInds ideal node count to prevent overfitting
def find_n_estimators(train_X, train_y, test_X, test_y):
	accuracy_forest_base = 0
	for i in range(10, 1000, 10):
		rf = RandomForestRegressor(random_state = 0, n_estimators = i)
		rf.fit(train_X, train_y)
		predictions_forest = rf.predict(test_X)
		for i in range(len(predictions_forest)):
			predictions_forest[i] = round(predictions_forest[i],0)
		accuracy_forest = accuracy_score(test_y, predictions_forest)
		if accuracy_forest > accuracy_forest_base:
			accuracy_forest_base = accuracy_forest
			n_est = i
		else:
			break
	return n_est

# FInding the accuracy of test set
n_est = find_n_estimators(train_X, train_y, test_X, test_y)
rf = RandomForestRegressor(random_state = 0, n_estimators = n_est)
rf.fit(train_X, train_y)
predictions_forest = rf.predict(test_X)
for i in range(len(predictions_forest)):
	predictions_forest[i] = round(predictions_forest[i],0)
accuracy_forest = accuracy_score(test_y, predictions_forest)
print(accuracy_forest)

rf_final = RandomForestRegressor(random_state = 0, n_estimators = n_est)
rf_final.fit(train_data[columns], train_data['Survived'])
test_predictions = rf_final.predict(test_data[columns])
for i in range(len(test_predictions)):
	test_predictions[i] = round(test_predictions[i],0)

# Organizing the submission data and outputting a csv
test_ids = test_data["PassengerId"]
output_df = {"PassengerId":test_ids, "Survived": test_predictions}
submission = pd.DataFrame(output_df)
submission.to_csv("submission.csv",index=False)
