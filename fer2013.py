import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Load GoEmotions data
# Load GoEmotions data
import pandas as pd
import os

# Load GoEmotions data
import pandas as pd
import os

# Load GoEmotions data
def load_goemotions_data(goemotions_path):
    print("Loading GoEmotions data...")
    texts = []
    labels = []
    
    for file in os.listdir(goemotions_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(goemotions_path, file))
            print(f"Loaded {file}, columns: {df.columns.tolist()}")  # Print the column names
            
            # Iterate through each row in the DataFrame
            for index, row in df.iterrows():
                if pd.notnull(row['text']):  # Ensure text is not null
                    texts.append(row['text'])  # Add the text to the texts list
                    
                    # Find the first emotion with a value of 1 to assign as the label
                    assigned_label = None
                    for emotion in df.columns[8:]:  # Adjust if necessary
                        if row[emotion] == 1:
                            assigned_label = emotion
                            break
                    if assigned_label:
                        labels.append(assigned_label)  # Append the assigned label
                    else:
                        labels.append('neutral')  # Assign a neutral label if no emotion found

    print(f"Loaded {len(texts)} texts and {len(labels)} labels.")
    print("GoEmotions data loaded successfully.")
    return texts, labels




# Load RAVDESS data
def load_ravdess_data(ravdess_path):
    print("Loading RAVDESS data...")
    texts = []  # Placeholder for actual audio file paths or features
    labels = []
    # Assuming labels are derived from the folder names or similar logic
    for actor_dir in os.listdir(ravdess_path):
        actor_path = os.path.join(ravdess_path, actor_dir)
        if os.path.isdir(actor_path):
            for audio_file in os.listdir(actor_path):
                if audio_file.endswith('.wav'):  # Adjust if using a different format
                    texts.append(os.path.join(actor_path, audio_file))  # Store path or extract features
                    labels.append(actor_dir)  # Assuming actor_dir represents the label
    print("RAVDESS data loaded successfully.")
    return texts, labels

# Load FER2013 data from train folder
def load_fer2013_data(fer2013_path):
    print("Loading FER2013 data...")
    images = []
    labels = []
    train_folder = os.path.join(fer2013_path, 'train')
    
    for img_file in os.listdir(train_folder):
        if img_file.endswith('.jpg'):  # Adjust if using a different format
            image_path = os.path.join(train_folder, img_file)
            images.append(image_path)  # Store image path
            # Extract label from filename or image metadata if applicable
            label = img_file.split('_')[0]  # Adjust logic to extract label
            labels.append(label)
    
    print("FER2013 data loaded successfully.")
    return images, labels

# Train and save models
def train_and_save_models(goemotions_path, ravdess_path, fer2013_path):
    # Load data
    texts_goemotions, labels_goemotions = load_goemotions_data(goemotions_path)
    texts_ravdess, labels_ravdess = load_ravdess_data(ravdess_path)
    texts_fer2013, labels_fer2013 = load_fer2013_data(fer2013_path)

    # Example pipeline
    model_goemotions = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    # Prepare data for training
    X_train_goemotions, X_test_goemotions, y_train_goemotions, y_test_goemotions = train_test_split(
        texts_goemotions, labels_goemotions, test_size=0.2, random_state=42)
    
    # Train model
    model_goemotions.fit(X_train_goemotions, y_train_goemotions)
    joblib.dump(model_goemotions, 'model_goemotions.joblib')

    # Process RAVDESS and FER2013 in similar way...

# Main execution
if __name__ == "__main__":
    goemotions_path = 'goemotion'  # Update with actual path
    ravdess_path = 'ravdess'  # Update with actual path
    fer2013_path = 'fer2013'  # Update with actual path
    train_and_save_models(goemotions_path, ravdess_path, fer2013_path)
