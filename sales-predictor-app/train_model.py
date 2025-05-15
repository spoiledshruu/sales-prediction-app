import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load data
df = pd.read_csv("advertising.csv")

# Split features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "linear_model.pkl")
print("✅ Model saved as linear_model.pkl")
