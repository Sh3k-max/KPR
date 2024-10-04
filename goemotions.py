import pandas as pd
from sklearnex import patch_sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump
import os

# Patch sklearn to use Intel optimizations
patch_sklearn()

def load_goemotions(dataset_path):
    # Load the first CSV file for training
    df = pd.read_csv(f'{dataset_path}/goemotions_1.csv')  # Adjust this line based on your files
    
    # Print columns to confirm the structure
    print("Columns in the dataset:")
    print(df.columns.tolist())
    
    return df

# Load dataset
df = load_goemotions('./goemotions')

# List of emotion columns
emotion_columns = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Create a directory to save models
os.makedirs('emotion_models', exist_ok=True)

# Loop through all emotion columns to train models
for emotion in emotion_columns:
    print(f"Training model for emotion: {emotion}")
    
    # Input feature (text)
    X = df['text']
    
    # Output label (current emotion)
    y = df[emotion]
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_vect, y_train)

    # Save the model and vectorizer
    model_filename = f'emotion_models/{emotion}_model.joblib'
    vectorizer_filename = f'emotion_models/{emotion}_vectorizer.joblib'
    
    dump(model, model_filename)
    dump(vectorizer, vectorizer_filename)

    print(f"Model and vectorizer for {emotion} saved successfully at {model_filename} and {vectorizer_filename}!")

print("All models trained and saved successfully!")
