import joblib
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model

class CombinedModel:
    def __init__(self, joblib_model_path_1, joblib_model_path_2, h5_model_path):
        # Load models
        self.model_joblib_1 = joblib.load(joblib_model_path_1)
        self.model_joblib_2 = joblib.load(joblib_model_path_2)
        self.model_h5 = load_model(h5_model_path)

    def predict(self, X):
        # Get predictions from each model
        preds_joblib_1 = self.model_joblib_1.predict(X)
        preds_joblib_2 = self.model_joblib_2.predict(X)
        preds_h5 = self.model_h5.predict(X)

        # Assuming predictions are probabilities or regression values, combine them
        # For classification, you might want to apply soft voting
        combined_preds = (preds_joblib_1 + preds_joblib_2 + preds_h5) / 3

        return combined_preds

    def save_combined_model(self, output_path):
        # Save the combined models to a single file
        combined_models = {
            'model_joblib_1': self.model_joblib_1,
            'model_joblib_2': self.model_joblib_2,
            'model_h5': self.model_h5
        }
        joblib.dump(combined_models, output_path)
        print(f'Combined model saved to {output_path}')

# Usage example
if __name__ == "__main__":
    # Specify your model paths here
    joblib_model_path_1 = 'emotion_model.joblib'
    joblib_model_path_2 = 'vectorizer.joblib'
    h5_model_path = 'fer2013_emotion_model.h5'
    
    # Create the combined model
    combined_model = CombinedModel(joblib_model_path_1, joblib_model_path_2, h5_model_path)

    # Example test data (replace with your actual data)
    # X_test = ...  # Your input data for predictions
    # final_predictions = combined_model.predict(X_test)

    # Optionally, save the combined model
    combined_model.save_combined_model('combined_models.joblib')
