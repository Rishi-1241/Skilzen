from flask import Flask
from flask_socketio import SocketIO, emit
import asyncio
from google.cloud import speech_v1p1beta1 as speech
import io
import pyaudio
import google.generativeai as genai
from google.cloud import speech
from google.cloud import texttospeech
import os
from pydub import AudioSegment
from pydub.playback import play
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from google.cloud import speech, texttospeech
import uuid
from google.cloud.dialogflowcx_v3 import AgentsClient, SessionsClient
from google.cloud.dialogflowcx_v3.types import session


GOOGLE_API_KEY = "AIzaSyCnP8cuR-cfpydBlHYbFv0fAtJhqdpHQKQ"
project_id = "certain-math-447716-d1"  # Your Google Cloud project ID
location_id = "global"  # Your agent's location, e.g., "global"
agent_id = "ab039e5f-d9ce-4feb-90ad-4184f23f01e5"  # Your Dialogflow CX agent ID
flow_id = "dd90ab06-761a-410d-bb04-f60368c323ac"
agent = f"projects/{project_id}/locations/{location_id}/agents/{agent_id}"

    # Use a unique session ID for the interaction
session_id = uuid.uuid4()
genai.configure(api_key=GOOGLE_API_KEY)
LANGUAGE_CONFIGS = {
    "English": {
        "code": "en-US",
        "voice_name": "en-US-Wavenet-F",
        "fallback_voice": "en-US-Standard-C"
    },
    "Hindi": {
        "code": "hi-IN",
        "voice_name": "hi-IN-Wavenet-A",
        "fallback_voice": "hi-IN-Standard-A"
    },
    "Telugu": {
        "code": "te-IN",
        "voice_name": "te-IN-Standard-A",
        "fallback_voice": "te-IN-Standard-A"
    },
    "Tamil": {
        "code": "ta-IN",
        "voice_name": "ta-IN-Wavenet-A",
        "fallback_voice": "ta-IN-Standard-A"
    },
    "Kannada": {
        "code": "kn-IN",
        "voice_name": "kn-IN-Wavenet-A",
        "fallback_voice": "kn-IN-Standard-A"
    },
    "Malayalam": {
        "code": "ml-IN",
        "voice_name": "ml-IN-Wavenet-A",
        "fallback_voice": "ml-IN-Standard-A"
    }
}

def get_dialogflow_response(text, language_code, agent, session_id, flow_id):
    """Get response from Dialogflow CX agent"""
    environment_id = "draft"  # Or use "production" if appropriate
    session_path = f"{agent}/environments/{environment_id}/sessions/{session_id}?flow={flow_id}"

    # Prepare text input for Dialogflow
    text_input = session.TextInput(text=text)
    query_input = session.QueryInput(text=text_input, language_code=language_code)

    # Create a detect intent request
    request = session.DetectIntentRequest(
        session=session_path,
        query_input=query_input,
    )

    # Create a session client
    session_client = SessionsClient()

    # Call the API
    response = session_client.detect_intent(request=request)

    # Get the response messages
    response_messages = [
        " ".join(msg.text.text) for msg in response.query_result.response_messages
    ]
    return " ".join(response_messages)

import os


import os
from pydub import AudioSegment
from pydub.utils import make_chunks
from scipy.io.wavfile import write
import numpy as np
import subprocess
import io
from google.cloud.speech_v1p1beta1 import types
def transcribe_audio(audio_file, language_code):
    print("Inside transcribe_audio")

    client = speech.SpeechClient()

    # Read the audio file as bytes
    audio_content = audio_file.read()

    # Prepare audio configuration
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        sample_rate_hertz=48000,
        language_code=language_code,
        model="latest_long",  # Optional: Choose the model based on your need
    )

    audio = speech.RecognitionAudio(content=audio_content)

    # Perform the transcription request
    print("Starting transcription...")
    start_time = time.time()  # Start latency measurement
    response = client.recognize(config=config, audio=audio)

    # Process the transcription response
    for result in response.results:
        if result:
            transcript = result.alternatives[0].transcript
            print(f"User said: {transcript}")
            end_time = time.time()  # End latency measurement
            latency_time = end_time - start_time
            print(f"Transcription latency: {latency_time:.2f} seconds")
            gemini_time = time.time()
            gemini_response = get_dialogflow_response(transcript, language_code, agent, session_id, flow_id)
            gemini_end_time = time.time()
            print(f"Gemini responds: {gemini_response}")
            gemini_audio_latency = gemini_end_time - gemini_time
            print(f"Gemini response latency: {gemini_audio_latency:.2f} seconds")

            yield gemini_response

import asyncio

response_audio_cache = {}

def synthesize_text_parallel(text, language):
    """Synthesize text to speech using WaveNet voices"""
    client = texttospeech.TextToSpeechClient()
    lang_config = LANGUAGE_CONFIGS[language]

    synthesis_input = texttospeech.SynthesisInput(text=text)
    try:
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang_config["code"],
            name=lang_config["voice_name"]
        )
    except Exception as e:
        print(f"WaveNet voice not available, falling back to standard voice: {str(e)}")
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang_config["code"],
            name=lang_config["fallback_voice"]
        )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        speaking_rate=1.0,
        pitch=0,
        volume_gain_db=0.0,
        effects_profile_id=["telephony-class-application"]
    )

    start_time = time.time()  # Start latency measurement
    try:
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        end_time = time.time()  # End latency measurement
        latency = end_time - start_time
        print(f"Speech synthesis latency: {latency:.2f} seconds")
        return response.audio_content
    except Exception as e:
        print(f"Error in speech synthesis: {str(e)}")
        voice.name = lang_config["fallback_voice"]
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        return response.audio_content


app = Flask(__name__)
from flask import Flask, request,jsonify
from flask_socketio import SocketIO, emit
from flask_socketio import SocketIO
from flask_cors import CORS

CORS(app)  # Enable CORS for the Flask app
socketio = SocketIO(app, cors_allowed_origins="*")
@app.route('/')
def index():
    return jsonify({"message": "SocketIO server is running"})
@socketio.on('transcribe')
def handle_transcribe(data):
    transcript = data.get('text')
    print(f"Transcribing: {transcript}")
    if transcript:
        response_text = get_dialogflow_response(transcript, "en-US", agent, session_id, flow_id)
        print(f"Dialogflow response: {response_text}")
        audio_content = synthesize_text_parallel(response_text, 'English')
        emit('audio_response', audio_content, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000,debug=True)