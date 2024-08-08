# print("Jello")
# import sys
# import json
# import numpy as np
# import pandas as pd
# import joblib

# # Load the model
# model = joblib.load('lung_cancer_model.pkl')

# # Parse input data
# input_data = json.loads(sys.argv[1])

# print(input_data)

# # Convert input data to DataFrame with feature names
# feature_names = [
#     'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
#     'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 
#     'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 
#     'SWALLOWING_DIFFICULTY', 'CHEST_PAIN', 'GENDER'
# ]
# input_df = pd.DataFrame([input_data], columns=feature_names)

# # Make prediction
# prediction = model.predict(input_df)

# # Print the prediction result
# print(prediction[0])

import sys
import pandas as pd
import joblib
import json
import numpy as np

# Example new data (replace this with your actual new data)
# Ensure it has the same structure and columns as your training data

input_data = json.loads(sys.argv[1])
print(input_data)

input_array = np.array([list(input_data.values())])

new_data = pd.DataFrame({
    'GENDER': [1],  # Male
    'AGE': [65],  # Example age
    'SMOKING': [1],  # Smoker
    'YELLOW_FINGERS': [1],  # Yellow fingers
    'ANXIETY': [1],  # Anxiety
    'PEER_PRESSURE': [1],  # Peer pressure
    'CHRONIC_DISEASE': [1],  # Chronic disease
    'FATIGUE': [1],  # Fatigue
    'ALLERGY': [0],  # No allergy
    'WHEEZING': [1],  # Wheezing
    'ALCOHOL_CONSUMING': [0],  # No alcohol consumption
    'COUGHING': [1],  # Coughing
    'SHORTNESS_OF_BREATH': [1],  # Shortness of breath
    'SWALLOWING_DIFFICULTY': [1],  # Swallowing difficulty
    'CHEST_PAIN': [1]  # Chest pain
})

model = joblib.load('lung_cancer_model.pkl')

# Predict using the trained model
predictions = model.predict(input_array)

# Print the predictions
print("Predictions:", predictions)

# To get the probability estimates for the positive class (lung cancer)
probabilities = model.predict_proba(input_array)[:, 1]
print("Probability of Lung Cancer:", probabilities)
