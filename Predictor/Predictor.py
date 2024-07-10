import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score

# Load data
matches = pd.read_csv("matches.csv", index_col=0)

# Data preprocessing
matches["date"] = pd.to_datetime(matches["date"])
matches["h/a"] = matches["venue"].astype("category").cat.codes  # Convert venue to a home (1) or away (0) number
matches["opp"] = matches["opponent"].astype("category").cat.codes  # Convert opponents to a number
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")  # Convert hours to number
matches["day"] = matches["date"].dt.dayofweek  # Convert day of week of game to a number
matches["target"] = (matches["result"] == "W").astype("int")  # Set a win to the value 1

# Train-test split
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']
predictors = ["h/a", "opp", "hour", "day"]

# Function to calculate rolling averages
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

# Apply rolling averages
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]
matches_rolling = matches.groupby("team", group_keys=False).apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling.index = range(matches_rolling.shape[0])

# Updated train-test split with rolling averages
train_rolling = matches_rolling[matches_rolling["date"] < '2022-01-01']
test_rolling = matches_rolling[matches_rolling["date"] > '2022-01-01']
predictors_rolling = predictors + new_cols

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'min_samples_split': [2, 5, 10],
    'max_depth': [None, 10, 20, 30]
}
rf = RandomForestClassifier(random_state=1)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='precision')
grid_search.fit(train_rolling[predictors_rolling], train_rolling["target"])

# Best parameters from GridSearchCV
best_rf = grid_search.best_estimator_

# Cross-validation score
cv_scores = cross_val_score(best_rf, train_rolling[predictors_rolling], train_rolling["target"], cv=5, scoring='precision')
print(f"Cross-validation precision scores: {cv_scores}")
print(f"Mean cross-validation precision: {np.mean(cv_scores)}")

# Fit the model on the training data
best_rf.fit(train_rolling[predictors_rolling], train_rolling["target"])

# Make predictions on the test data
preds_rolling = best_rf.predict(test_rolling[predictors_rolling])

# Evaluate the model
acc_rolling = accuracy_score(test_rolling["target"], preds_rolling)
precision_rolling = precision_score(test_rolling["target"], preds_rolling)
conf_matrix = confusion_matrix(test_rolling["target"], preds_rolling)

print(f"Accuracy: {acc_rolling}")
print(f"Precision: {precision_rolling}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_report(test_rolling["target"], preds_rolling))

# Display combined results
combined_rolling = pd.DataFrame(dict(actual=test_rolling["target"], prediction=preds_rolling))
combined_rolling = combined_rolling.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
print(combined_rolling)

# Class for handling missing dictionary keys
class MissingDict(dict):
    __missing__ = lambda self, key: key

# Mapping team names
map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)

combined_rolling["new_team"] = combined_rolling["team"].map(mapping)

# Merging predictions for both home and away teams
merged = combined_rolling.merge(combined_rolling, left_on=["date", "new_team"], right_on=["date", "opponent"])
print(merged)
