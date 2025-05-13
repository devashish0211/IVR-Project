
import streamlit as st
import openai
import whisper
import os
import tempfile

# Title
st.title("🎙️ AI-Powered IVR Intent Detector")

# Set your OpenAI API Key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Upload audio file
audio_file = st.file_uploader("Upload your voice file (.mp3 or .wav)", type=["mp3", "wav"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_file_path = tmp_file.name

    st.audio(audio_file, format='audio/wav')

    # Load Whisper model
    st.info("Transcribing using Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(tmp_file_path)
    transcript = result["text"]
    st.success("Transcription:")
    st.write(transcript)

    # Detect intent using GPT
    st.info("Detecting Intent with GPT...")
    prompt = f'''
    The customer said: "{transcript}"
    What is the user's intent? Choose one of: billing, order_status, technical_support, speak_to_agent, unknown.
    Respond with only the intent label.
    '''
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    intent = response["choices"][0]["message"]["content"].strip().lower()
    st.success(f"✅ Detected Intent: **{intent}**")

    # Route to department
    def route(intent):
        routes = {
            "billing": "📞 Routing to Billing Department...",
            "order_status": "📦 Routing to Order Status...",
            "technical_support": "🛠️ Routing to Technical Support...",
            "speak_to_agent": "👨‍💼 Connecting you to a Human Agent...",
            "unknown": "❓ Sorry, we could not understand your request."
        }
        return routes.get(intent, routes["unknown"])

    st.write(route(intent))
