import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

# Function to load FER2013 data
def load_fer2013_data(dataset_path):
    images = []
    labels = []
    
    # Load training data
    train_dir = os.path.join(dataset_path, 'train')
    for emotion in os.listdir(train_dir):
        emotion_dir = os.path.join(train_dir, emotion)
        if os.path.isdir(emotion_dir):
            for img_file in os.listdir(emotion_dir):
                img_path = os.path.join(emotion_dir, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Warning: Unable to read image file: {img_path}")
                    continue
                image = cv2.resize(image, (48, 48))  # Resize to 48x48
                images.append(image)
                labels.append(emotion)  # Use the folder name as the label

    return np.array(images), np.array(labels)

# Load FER2013 data
X_fer, y_fer = load_fer2013_data('./fer2013')

# Normalize and reshape images
X_fer = X_fer.reshape(-1, 48, 48, 1) / 255.0  # Reshape for CNN and normalize pixel values

# Convert string labels to numerical values
unique_labels = np.unique(y_fer)
label_to_num = {label: index for index, label in enumerate(unique_labels)}
y_fer_num = np.array([label_to_num[label] for label in y_fer])

# One-hot encoding of labels
y_fer_encoded = to_categorical(y_fer_num)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_fer, y_fer_encoded, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(unique_labels), activation='softmax')  # Output layer with the number of emotions
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Save the model
model.save('fer2013_emotion_model.h5')
