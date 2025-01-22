from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import base64
import asyncio
import io
from pydub import AudioSegment
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
async def transcribe_streaming_parallel(audio_data, language_code):
    """Transcribe streaming audio with latency measurement."""
    print("Transcribing audio")

    # Ensure the audio data is in the correct format
    audio_stream = io.BytesIO(audio_data)  # Convert the audio data to a BytesIO stream

    client = speech.SpeechClient()
    print("Client created")

    # Set the recognition config (Exotel uses 8kHz, 16-bit mono PCM)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  # PCM format
        sample_rate_hertz=8000,  # 8kHz sample rate (Exotel audio)
        language_code=language_code,
        model="latest_long",  # Adjust model based on your need
    )
    print("Config created")

    # Set streaming recognition config
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=False  # You can set this to True if you need interim results
    )

    # Increase chunk size to process more audio at once
    def generate_audio_chunks(stream, chunk_size=960):  # Increase chunk size
        """Yield chunks of audio content."""
        while chunk := stream.read(chunk_size):
            if chunk:
                print(f"Audio chunk size: {len(chunk)}")  # Log chunk size for debugging
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            else:
                break

    requests = generate_audio_chunks(audio_stream)

    print("Requests created")

    try:
        responses = client.streaming_recognize(streaming_config, requests)
        print("Responses received")

        # Logging the responses for debugging
        for response in responses:
            print(f"Response: {response}")
            for result in response.results:
                if result.is_final:
                    start_time = time.time()  # Start latency measurement for transcription
                    transcript = result.alternatives[0].transcript
                    end_time = time.time()  # End latency measurement for transcription

                    transcription_latency = end_time - start_time
                    print(f"User said: {transcript}")
                    print(f"Transcription latency: {transcription_latency:.2f} seconds")

                    # Assuming 'get_dialogflow_response' processes the transcript with Dialogflow
                    gemini_response = get_dialogflow_response(transcript, language_code, agent, session_id, flow_id)
                    print(f"Gemini responds: {gemini_response}")

                    yield gemini_response

    except Exception as e:
        print(f"Error during streaming recognition: {str(e)}")





import asyncio

response_audio_cache = {}

def synthesize_text_parallel(text, language):
    """Synthesize text to speech using WaveNet voices"""
    client = texttospeech.TextToSpeechClient()
    lang_config = LANGUAGE_CONFIGS["English"]

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

import base64
import io
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketState
import uvicorn

app = FastAPI()

@app.get("/")
async def home():
    return "Exotel Voicebot WebSocket Server Running"
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    try:
        while websocket.client_state == WebSocketState.CONNECTED:
            data = await websocket.receive_json()
            
            if "media" in data:
                try:
                    # Decode base64 audio from Exotel (slin format)
                    audio_data = base64.b64decode(data["media"]["payload"])
                    print(f"Audio data length: {len(audio_data)}")
                    audio_file = "output.mp3"

                    # Save the audio data as an MP3 file
                    with open(audio_file, "wb") as file:
                        file.write(audio_data)
                    
                    print("Audio data received and decoded")

                    # Process STT
                    async def process_audio():
                        lang = "English"  # Specify language
                        print(f"Processing audio in {lang}")

                        # Transcribe the audio
                        async for response_text in transcribe_streaming_parallel(audio_data, "en-US"):
                            print(f"Model said: {response_text}")

                            # Generate response from Dialogflow or model
                            gemini_response = response_text  # Replace with your model call

                            # TTS
                            print("Synthesizing speech")
                            audio_content = synthesize_text_parallel(gemini_response, "en-US")
                            print("Audio synthesized")
    
                            # Save the audio content as an MP3 file
                            audio_file_path = "response.mp3"
                            with open(audio_file_path, "wb") as audio_file:
                                audio_file.write(audio_content)  # Writing the synthesized audio content to file
                            print(f"Audio saved as {audio_file_path}")

                            # Encode audio back to base64
                            encoded_audio = base64.b64encode(audio_content).decode("utf-8")
                            print("Sending audio response")
                        
                            # Send the audio response via WebSocket
                            await websocket.send_json({"event": "media", "media": {"payload": encoded_audio}})
                            print("Audio response sent")

                    await process_audio()
                except Exception as e:
                    print(f"Error processing media: {e}")
                    await websocket.send_json({"event": "error", "message": str(e)})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Unexpected error: {e}")
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
