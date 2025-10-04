import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

data = {
    "rainfall": [100, 200, 150, 300, 250],
    "temperature": [20, 25, 23, 30, 28],
    "soil_quality": [7, 8, 6, 9, 7],
    "yield": [2.5, 3.0, 2.7, 3.5, 3.2]
}

df = pd.DataFrame(data)

X = df[["rainfall", "temperature", "soil_quality"]]
y = df["yield"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

joblib.dump(model, "crop_yield_model.pkl")

print("✅ Model trained and saved as crop_yield_model.pkl")
