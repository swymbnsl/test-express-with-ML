import sys
import pandas as pd
import joblib
import json
import numpy as np

input_data = json.loads(sys.argv[1])
print(input_data)

input_array = np.array([list(input_data.values())])

model = joblib.load('lung_cancer_model.pkl')

# Predict using the trained model
predictions = model.predict(input_array)

# Print the predictions
print("Predictions:", predictions)

# To get the probability estimates for the positive class (lung cancer)
probabilities = model.predict_proba(input_array)[:, 1]
print("Probability of Lung Cancer:", probabilities)
