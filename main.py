import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import string
from tensorflow import keras
from gtts import gTTS
from io import BytesIO

# --- Model and MediaPipe Setup ---

# Cache the Keras model to avoid reloading on each run
@st.cache_resource
def load_model_isl():
    """Loads the pre-trained ISL model."""
    try:
        return keras.models.load_model("model_isl_fixed.h5")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure 'model_isl_fixed.h5' is in the same directory.")
        return None

# Load the model once
model = load_model_isl()

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define the character set for ISL (A-Z and 1-9)
isl_alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

# --- Session State Initialization ---
# Persist variables across reruns

if "sentence" not in st.session_state:
    st.session_state.sentence = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "count" not in st.session_state:
    st.session_state.count = 0
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "voice_language" not in st.session_state:
    st.session_state.voice_language = "English"

# --- Utility Functions ---

def calc_landmark_list(image, landmarks):
    """Converts MediaPipe landmarks to a list of pixel coordinates."""
    image_width, image_height = image.shape[1], image.shape[0]
    return [[int(lm.x * image_width), int(lm.y * image_height)] for lm in landmarks.landmark]

def pre_process_landmark(landmarks):
    """Normalizes landmarks to make them translation and scale invariant."""
    base_x, base_y = landmarks[0]
    relative_landmarks = [[x - base_x, y - base_y] for x, y in landmarks]
    flat_landmarks = np.array(relative_landmarks).flatten()
    
    max_val = np.max(np.abs(flat_landmarks))
    if max_val == 0:
        max_val = 1
        
    normalized_landmarks = flat_landmarks / max_val
    return normalized_landmarks.tolist()

def predict_sign(landmark_vector):
    """Predicts the sign character from the landmark vector."""
    if model is None:
        return ""
    df = pd.DataFrame([landmark_vector])
    prediction = model.predict(df, verbose=0)
    predicted_char_index = np.argmax(prediction)
    return isl_alphabet[predicted_char_index]

def text_to_speech(text, lang_code):
    """Generates audio from text using gTTS and returns it as bytes."""
    if not text.strip():
        st.warning("There is no text to speak.")
        return None
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp.read()
    except Exception as e:
        st.error(f"Could not generate audio: {e}")
        return None

# --- Streamlit UI ---

def main():
    """Main function to run the Streamlit app."""
    st.title("🙌👌 MUDRA - ISL Translator")
    st.sidebar.header("Settings ⚙️")

    # Optional: Display a logo in the sidebar with a fallback
    try:
        st.sidebar.image("Poster_Logo.png", use_container_width=True)
    except Exception:
        st.sidebar.warning("Logo image not found.")

    # --- Controls in Sidebar ---
    st.sidebar.markdown("---")
    st.session_state.voice_language = st.sidebar.selectbox(
        "Select Voice Language",
        ("English", "Hindi", "Marathi"),
        key="lang_select"
    )
    
    # --- Separate Start and Stop Buttons in columns ---
    start_col, stop_col = st.sidebar.columns(2)

    if start_col.button("Start Camera", key="start_cam", use_container_width=True, disabled=st.session_state.camera_active):
        st.session_state.camera_active = True
        if 'rerun' in dir(st):
            st.rerun()

    if stop_col.button("Stop Camera", key="stop_cam", use_container_width=True, disabled=not st.session_state.camera_active):
        st.session_state.camera_active = False
        # Clear sentence and predictions when camera is stopped manually
        st.session_state.sentence = []
        st.session_state.last_prediction = None
        st.session_state.count = 0
        if 'rerun' in dir(st):
            st.rerun()
    
    audio_placeholder = st.sidebar.empty()
    st.sidebar.markdown("---")

    st.write("Translate Indian Sign Language in real-time. Use the buttons in the sidebar to control the app.")

    # Layout for video feed and static image
    col1, col2 = st.columns([2, 1])
    frame_placeholder = col1.empty()
    
    with col2:
        try:
            st.image("Poster_isl.png", caption="ISL Alphabet Chart")
        except Exception:
            st.warning("ISL poster not found.")

    # Placeholder for the translated sentence
    sentence_placeholder = st.empty()
    sentence_placeholder.markdown(f"### Sentence: `{''.join(st.session_state.sentence)}`")

    # --- Sentence Control Buttons ---
    st.write("---")
    # Updated to 4 columns to include the new "Speak" button
    col3, col4, col5, col6 = st.columns(4)
    
    if col3.button("Add Space", use_container_width=True):
        st.session_state.sentence.append(" ")
        st.session_state.last_prediction = " "

    if col4.button("Backspace", use_container_width=True) and st.session_state.sentence:
        st.session_state.sentence.pop()
        st.session_state.last_prediction = None

    if col5.button("Clear All", use_container_width=True):
        st.session_state.sentence = []
        st.session_state.last_prediction = None
    
    # New "Speak" button to say the whole sentence
    if col6.button("Speak", use_container_width=True):
        full_sentence_str = "".join(st.session_state.sentence).strip()
        lang_map = {"English": "en", "Hindi": "hi", "Marathi": "mr"}
        lang_code = lang_map[st.session_state.voice_language]
        audio_bytes = text_to_speech(full_sentence_str, lang_code)
        if audio_bytes:
            audio_placeholder.audio(audio_bytes, format="audio/mp3", autoplay=True)


    # --- Main Application Loop ---
    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)
        with mp_hands.Hands(
            model_complexity=0, max_num_hands=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        ) as hands:
            while st.session_state.camera_active and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("Could not read frame from camera.")
                    break

                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmark_list = calc_landmark_list(frame, hand_landmarks)
                        processed_landmarks = pre_process_landmark(landmark_list)
                        prediction = predict_sign(processed_landmarks)

                        if prediction == st.session_state.last_prediction:
                            st.session_state.count += 1
                        else:
                            st.session_state.count = 1
                        st.session_state.last_prediction = prediction

                        if st.session_state.count >= 10:
                            if not st.session_state.sentence or st.session_state.sentence[-1] != prediction:
                                st.session_state.sentence.append(prediction)
                            st.session_state.count = 0

                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        cv2.putText(frame, prediction, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

                frame_placeholder.image(frame, channels="BGR")
                sentence_placeholder.markdown(f"### Sentence: `{''.join(st.session_state.sentence)}`")
                
        cap.release()
    else:
        frame_placeholder.markdown("### Camera is off.\nPress **Start Camera** in the sidebar to begin.")

if __name__ == "__main__":
    main()
