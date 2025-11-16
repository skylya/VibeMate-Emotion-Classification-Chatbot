import streamlit as st
import torch
import random
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ----------------------------------------------------
# Load Model + Tokenizer
# ----------------------------------------------------
save_directory = "./emotion_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(save_directory)
model = DistilBertForSequenceClassification.from_pretrained(save_directory)

# ----------------------------------------------------
# Label Mapping
# ----------------------------------------------------
label2id = model.config.label2id
id2label = {v: k for k, v in label2id.items()}

# ----------------------------------------------------
# Emotion-to-song mapping
# ----------------------------------------------------
emotion_map = {
    "joy": [
        ("Shake It Off - Taylor Swift", "Keep shining and enjoy every moment!"),
        ("Love Yourself - Justin Bieber", "Celebrate the little things today!"),
        ("Shape of You - Ed Sheeran", "Let your happiness shine through!")
    ],
    "sad": [
        ("Someone You Loved - Ed Sheeran", "It's okay to feel sad, this too shall pass."),
        ("Stay - Justin Bieber", "Take your time to heal."),
        ("All Too Well - Taylor Swift", "Feel the emotions, let it out.")
    ],
    "anger": [
        ("Bad Blood - Taylor Swift", "Take a deep breath and let go of anger."),
        ("You Need Me, I Don’t Need You - Ed Sheeran", "Let the beat help you release frustration."),
        ("Look What You Made Me Do - Taylor Swift", "Take a deep breath and let go of anger.")
    ],
    "fear": [
        ("Fearless - Taylor Swift", "Stay strong, everything will be okay."),
        ("Hold On - Justin Bieber", "Face your fears one step at a time."),
        ("Thinking Out Loud - Ed Sheeran", "You’ve got the strength to overcome this.")
    ],
    "love": [
        ("Perfect - Ed Sheeran", "Love yourself and cherish the moments."),
        ("Lover - Taylor Swift", "Express love and enjoy every heartbeat."),
        ("Intentions - Justin Bieber", "Love comes in many forms, embrace it!")
    ],
    "suprise": [
        ("Can’t Stop the Feeling! - Justin Timberlake", "Expect the unexpected and enjoy the ride!"),
        ("22 - Taylor Swift", "Life is full of surprises, enjoy them!"),
        ("Sing - Ed Sheeran", "Be ready for anything today!")
    ]
}

# ----------------------------------------------------
# Streamlit Setup
# ----------------------------------------------------
st.set_page_config(page_title="VibeMate Chatbot", page_icon="🎧", layout="centered")

st.title("🎧 VibeMate – Emotion Classification Chatbot")
st.write("Chat with the emotion-aware music recommender 🤖🎶")

# ----------------------------------------------------
# Initialize chat history
# ----------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ----------------------------------------------------
# Sidebar: Model & App Info
# ----------------------------------------------------
st.sidebar.header("📌 Model Information")
st.sidebar.write("**Model:** DistilBERT-base-uncased")
st.sidebar.write(f"**Number of Labels:** {len(label2id)}")
st.sidebar.write("**Emotion Labels:**")
for label in id2label.values():
    st.sidebar.write(f"- {label}")

st.sidebar.header("📁 About")
st.sidebar.info(
    "VibeMate is an emotion-aware chatbot that detects the user's emotion and "
    "recommends a suitable song + motivational quote."
)

st.sidebar.caption("Built with ❤️ using Streamlit + Transformers")


# ----------------------------------------------------
# Text Input + Prediction
# ----------------------------------------------------
user_input = st.text_input("Type a message:")

if st.button("Send"):
    if user_input.strip() != "":
        # Tokenize
        encoding = tokenizer(user_input, truncation=True, padding=True, return_tensors="pt")

        # Predict emotion
        outputs = model(**encoding)
        pred_label = torch.argmax(outputs.logits, dim=1).item()
        emotion = id2label[pred_label]

        # Pick song + quote
        song, quote = random.choice(emotion_map.get(emotion, [("","")]))

        # Save to chat history
        st.session_state.chat_history.append({
            "user": user_input,
            "emotion": emotion,
            "song": song,
            "quote": quote
        })


# ----------------------------------------------------
# Display Chat History
# ----------------------------------------------------
st.write("### 💬 Chat History")

if len(st.session_state.chat_history) == 0:
    st.info("No messages yet. Start chatting above!")
else:
    for chat in reversed(st.session_state.chat_history):  # latest first
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Emotion:** `{chat['emotion']}`")
        st.markdown(f"**🎵 Song Recommendation:** {chat['song']}")
        st.markdown(f"**💬 Quote:** *{chat['quote']}*")
        st.write("---")

# Footer
st.caption("Powered by DistilBERT • Built with Streamlit 🚀")
