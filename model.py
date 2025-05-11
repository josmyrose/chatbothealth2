
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

# Load your dataset
df = pd.read_csv('symptoms.csv')

# Preprocessing
# Fill NaN values with 0 (assuming missing symptoms mean the symptom isn't present)
df = df.fillna(0)

# Separate features (symptoms) and target (diseases)
X = df.drop('diseases', axis=1)
y = df['diseases']

# Train a simple model (for demonstration)
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
joblib.dump(model, 'disease_predictor.joblib')

# Create a symptom list for the chatbot
symptoms = list(X.columns)
with open('symptoms.txt', 'w') as f:
    f.write('\n'.join(symptoms))