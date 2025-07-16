import streamlit as st
import pickle

# Load the model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Define suggestions based on emotion
suggestions = {
    "joy": "😊 Great to hear! Keep spreading positivity!",
    "sadness": "😢 It's okay to feel sad. Try talking to a friend or taking a walk.",
    "anger": "😠 Take a deep breath. Try some relaxation exercises.",
    "fear": "😨 You're safe. Reach out if you need support.",
    "love": "❤️ Love is beautiful! Keep it kind and strong.",
    "surprise": "😲 That was unexpected! Stay curious.",
}

# App Title
st.set_page_config(page_title="Mental Health Chat Analyzer")
st.title("🧠 Mental Health Chat Analyzer")
st.markdown("Enter a message below to analyze its emotional tone.")

# Input
user_input = st.text_area("Type a chat message here:", height=100)

# Process
if st.button("Analyze Emotion"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        # Vectorize and predict
        vect_input = vectorizer.transform([user_input])
        prediction = model.predict(vect_input)[0]

        # Display result
        st.success(f"**Predicted Emotion:** {prediction.capitalize()}")
        st.markdown(f"**Suggestion:** {suggestions.get(prediction, '🙂 Stay strong and positive!')}")



