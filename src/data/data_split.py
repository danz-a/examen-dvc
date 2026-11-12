import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["split"]

df = pd.read_csv("data/raw_data/raw.csv").drop(columns="date")
X = df.drop(columns="silica_concentrate")
y = df["silica_concentrate"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"])

for name, data in {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}.items():
    data.to_csv(f"data/processed_data/{name}.csv", index=False)
