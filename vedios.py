import os
import librosa
import numpy as np
import pandas as pd
from sklearnex import patch_sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Initialize Intel Extension for scikit-learn
patch_sklearn()

# Path to RAVDESS dataset
data_dir = 'model'  # Change this to your dataset path

# List to store data
data = []

# Load each file and extract MFCC features
for filename in os.listdir(data_dir):
    if filename.endswith('.wav'):
        # Load audio file
        file_path = os.path.join(data_dir, filename)
        signal, sr = librosa.load(file_path, sr=None)
        
        # Extract MFCC features
        mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).T, axis=0)
        
        # Get emotion label from the filename
        label = filename.split('-')[2]  # Adjust based on your filename structure
        
        # Append to data
        data.append((mfccs, label))

# Create DataFrame
df = pd.DataFrame(data, columns=['mfccs', 'label'])

# Convert mfccs into a 2D array
X = np.array(df['mfccs'].tolist())
y = df['label'].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save the model
joblib.dump(model, 'emotion_recognition_model.pkl')
