import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

# Load the dataset
df = pd.read_csv("data/StudentsPerformance.csv")

# Features and target variable
X = df[["gender", "race/ethnicity", "parental level of education", "test preparation course"]]
y = df["math score"]

# Preprocessing: One-hot encode categorical columns
preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown='ignore'), X.columns)
], remainder="passthrough")

# Define pipeline: preprocessing + model
model_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
model_pipeline.fit(X, y)

# Save trained model
joblib.dump(model_pipeline, "student_model.pkl")
print("✅ Model trained and saved as 'student_model.pkl'")
