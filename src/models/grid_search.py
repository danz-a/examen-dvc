import joblib
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

param_grid = yaml.safe_load(open("params.yaml"))["gridsearch"]

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").squeeze("columns")

grid = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring="r2")
grid.fit(X_train, y_train)
joblib.dump(grid.best_params_, "models/best_params.pkl")
