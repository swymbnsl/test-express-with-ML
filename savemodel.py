import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
dataset = pd.read_excel(r"C:\Users\swaya\OneDrive\Desktop\Python\LungModel\LUNG CANCER DATASET.xlsx")

# Drop missing values
dataset = dataset.dropna()

# Encode categorical variables
dataset['GENDER'] = dataset['GENDER'].replace({'M': 1, 'F': 0})
dataset['LUNG_CANCER'] = dataset['LUNG_CANCER'].replace({'NO': 0, 'YES': 1})

# Split the data into features and target
X = dataset.drop('LUNG_CANCER', axis=1)
y = dataset['LUNG_CANCER']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=30)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'lung_cancer_model.pkl')

# Optional: Print model scores
print(f'Training Score: {model.score(X_train, y_train)}')
print(f'Testing Score: {model.score(X_test, y_test)}')