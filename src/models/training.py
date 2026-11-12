import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").squeeze("columns")

model = GradientBoostingRegressor(random_state=42, **joblib.load("models/best_params.pkl"))
model.fit(X_train, y_train)
joblib.dump(model, "models/gbr_model.pkl")
