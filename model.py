import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load data
df = pd.read_csv("accidents_dataset.csv")

# Feature engineering
X = pd.get_dummies(df.drop('accident_risk', axis=1))
y = df['accident_risk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, predictions):.2f}")
print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")

# Save model
joblib.dump(model, "accident_model.pkl")
print("Model saved as accident_model.pkl")