import streamlit as st
import ollama
import joblib  # Importing joblib to load your model
import logging

# Suppress logging from the ollama library and Streamlit's logger
logging.getLogger("ollama").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# ==========================
# Load Your Emotion Identification Model
# ==========================
@st.cache_resource  # Caching the model to speed up future calls
def load_model():
    model = joblib.load('model.joblib')
    return model

model = load_model()

@st.cache_data  # Cache predictions for the same input
def identify_emotion(text):
    predictions = model.predict([text])
    
    if isinstance(predictions[0], str):
        return predictions[0]  # If the model returns the label directly

    emotions = ['happy', 'sad', 'angry', 'neutral']  # Map of emotions based on your model's output

    index = int(predictions[0])  # Ensure this is converted to an integer
    return emotions[index]  # Return the emotion corresponding to the index

def get_emotion_aware_prompt(user_input, emotion):
    emotion_context = {
        'happy': "The user is feeling happy and satisfied.",
        'sad': "The user is feeling sad and might need encouragement.",
        'angry': "The user is feeling angry and needs a calm and empathetic response.",
        'neutral': "The user has a neutral tone."
    }

    context = emotion_context.get(emotion, "The user is expressing their feelings.")

    prompt = (
        f"You are an empathetic customer service chatbot. {context} "
        f"Respond appropriately to assist them.\n\n"
        f"User: {user_input}\n"
        f"Chatbot:"
    )
    return prompt

@st.cache_data  # Cache the responses for the same prompt
def get_response_from_ollama(prompt):
    try:
        response = ollama.generate(prompt=prompt, model="llama3.1")
        return response['response']
    except Exception as e:
        return "I'm sorry, I'm having trouble processing your request right now."

# ==========================
# Custom CSS for Styling
# ==========================
st.markdown(
    """
    <style>
        body {
            background-image: url('https://cdn.prod.website-files.com/5e42772e6a8cfd42a9715206/62f22f3eb2506eb175b32a1b_Article_Emotional-Chatbot-_1_.jpeg');
            background-size: cover;
            color: white;
        }
        .front-page {
            text-align: center;
            padding: 100px;
            background-color: rgba(0, 0, 0, 0.5);
            border-radius: 10px;
        }
        .chat-title {
            font-size: 2.5em;
            margin-bottom: 20px;
        }
        .chat-description {
            font-size: 1.2em;
            margin-bottom: 40px;
        }
        .button {
            font-size: 1.2em;
            padding: 10px 20px;
            color: white;
            background-color: #007BFF;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .button:hover {
            background-color: #0056b3;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# Streamlit Application
# ==========================
def main():
    st.title("Emotion-Aware Customer Service Chatbot")

    # Front Page
    if st.session_state.page == "front":
        st.markdown(
            """
            <div class="front-page">
                <h1 class="chat-title">Welcome to the Emotion-Aware Chatbot!</h1>
                <p class="chat-description">This chatbot uses advanced emotion recognition to provide empathetic customer service.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if st.button("Start Chat"):
            st.session_state.page = "chat"

    # Chat Page
    elif st.session_state.page == "chat":
        st.markdown(
            """
            <div class="front-page">
                <h1 class="chat-title">Chat with the Bot!</h1>
                <p class="chat-description">Type your message below and press 'Enter' to chat with the bot.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        user_input = st.text_input("You:", "")

        if user_input:
            emotion = identify_emotion(user_input)
            st.write(f"(Debug) Detected Emotion: {emotion}")

            prompt = get_emotion_aware_prompt(user_input, emotion)

            # Use a placeholder for output without showing any default running messages
            with st.spinner("Processing..."):
                chatbot_response = get_response_from_ollama(prompt)

            # Display the chatbot's response after processing is complete
            st.write(f"Chatbot: {chatbot_response}")

# ==========================
# Entry Point
# ==========================
if __name__ == "__main__":
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'front'
    
    main()
