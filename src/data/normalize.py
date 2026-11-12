import pandas as pd
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv("data/processed_data/X_train.csv")
X_test = pd.read_csv("data/processed_data/X_test.csv")

scaler = StandardScaler().fit(X_train)
pd.DataFrame(scaler.transform(X_train), columns=X_train.columns).to_csv("data/processed_data/X_train_scaled.csv", index=False)
pd.DataFrame(scaler.transform(X_test), columns=X_test.columns).to_csv("data/processed_data/X_test_scaled.csv", index=False)
