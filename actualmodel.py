import joblib
import numpy as np

# Load the combined model
model_path = 'model_goemotions.joblib'  # Update with the correct path
model = joblib.load(model_path)

# Define a function to get the predicted emotion
def predict_emotion(text):
    # Convert text to the required format for prediction
    predictions = model.predict([text])
    return predictions

# Function to create responses based on predicted emotion
def generate_response(emotion):
    responses = {
        'admiration': "Thank you! I'm glad you feel that way. How can I assist you further?",
        'amusement': "Haha! I'm glad you're enjoying this. How can I help you?",
        'anger': "I understand that you're upset. Let's see how we can resolve this.",
        'annoyance': "I see this is frustrating for you. Let’s work on finding a solution.",
        'approval': "Thank you for your approval! How can I assist you today?",
        'caring': "That's very kind of you! How can I help you?",
        'confusion': "I understand that this might be confusing. What can I clarify for you?",
        'curiosity': "Great question! I'm here to provide you with the information you need.",
        'desire': "I see you're eager! Let’s get you what you’re looking for.",
        'disappointment': "I'm sorry to hear that. How can we make things better?",
        'disapproval': "I understand you might not be satisfied. What can we improve?",
        'disgust': "I see this is not to your liking. How can I assist you further?",
        'embarrassment': "No need to feel embarrassed! I'm here to help.",
        'excitement': "That's awesome! I'm excited to help you with anything you need.",
        'fear': "I understand that this might be scary. Let’s address your concerns.",
        'gratitude': "You're welcome! I'm here for you. How can I assist you today?",
        'grief': "I'm really sorry to hear that. If there's anything I can do, please let me know.",
        'joy': "That's wonderful to hear! How can I assist you today?",
        'love': "Thank you! I appreciate your kindness. What can I do for you?",
        'nervousness': "It's okay to feel nervous. I'm here to help ease your worries.",
        'optimism': "It's great to see a positive outlook! How can I support you?",
        'pride': "That's fantastic! I'm proud to assist you. What can I help with?",
        'realization': "It's great that you've come to this understanding. How can I assist you further?",
        'relief': "I'm glad to hear you're feeling relieved! What else can I help you with?",
        'remorse': "I understand how you feel. How can I assist you in moving forward?",
        'sadness': "I'm here for you. How can I help you?",
        'surprise': "Oh! That’s interesting! How can I assist you with that?",
        'neutral': "How can I assist you today?",
    }
    return responses.get(emotion, "I'm here to help. Please tell me more.")

# Chatbot interaction loop
def chat():
    print("Welcome to the Emotion-Aware Customer Care Chatbot!")
    print("Type 'exit' to end the chat.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Chatbot: Thank you for chatting! Goodbye!")
            break
        
        # Predict the emotion based on user input
        predicted_emotion = predict_emotion(user_input)
        
        # Generate a response based on the predicted emotion
        response = generate_response(predicted_emotion[0])  # Get the first prediction
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()
