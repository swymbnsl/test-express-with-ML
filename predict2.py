import sys
import json
import numpy as np
import joblib

print("helllo")

try:
    # Load the model
    print("Loading model...")
    model = joblib.load('lung_cancer_model.pkl')
    print("Model loaded")

    # Parse input data
    print("Parsing input data...")
    input_data = json.loads(sys.argv[1])
    input_array = np.array([list(input_data.values())])
    print("Input data parsed:", input_array)

    # Make prediction
    print("Making prediction...")
    prediction = model.predict(input_array)
    print("Prediction made:", prediction)

    # Print the prediction result
    print(prediction[0])
except Exception as e:
    print("Error occurred:", e)
