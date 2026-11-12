import json
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

model = joblib.load("models/gbr_model.pkl")
X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv").squeeze("columns")

y_pred = model.predict(X_test)
pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv("data/prediction.csv", index=False)
with open("metrics/scores.json", "w") as f:
    json.dump({"mse": mean_squared_error(y_test, y_pred), "r2": r2_score(y_test, y_pred)}, f, indent=4)
