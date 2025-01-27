import asyncio
import base64
import json
import time
from google.cloud import speech
from pydub import AudioSegment
import io
import websockets
import queue
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
import asyncio
import base64
import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google.cloud import speech
import queue
from pydub import AudioSegment
import io

app = FastAPI()

# Assuming you have these variables or constants set up:
# `LANGUAGE_CONFIGS`, `get_dialogflow_response`, `synthesize_text_parallel`, and `audio_stream_generator`
GOOGLE_API_KEY = "AIzaSyCnP8cuR-cfpydBlHYbFv0fAtJhqdpHQKQ"
project_id = "certain-math-447716-d1"  # Your Google Cloud project ID
location_id = "global"  # Your agent's location, e.g., "global"
agent_id = "ab039e5f-d9ce-4feb-90ad-4184f23f01e5"  # Your Dialogflow CX agent ID
flow_id = "dd90ab06-761a-410d-bb04-f60368c323ac"
agent = f"projects/{project_id}/locations/{location_id}/agents/{agent_id}"

    # Use a unique session ID for the interaction



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

class AudioStreamHandler:
    def __init__(self):
        self.buff = queue.Queue()

    def fill_buffer(self, in_data):
        """Continuously collect data from the audio stream, into the buffer."""
        self.buff.put(in_data)

    def generator(self):
        """Generate chunks of audio data from the buffer."""
        while True:
            # Ensure there's at least one chunk of data in the buffer.
            chunk = self.buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Process any additional data in the buffer.
            while True:
                try:
                    chunk = self.buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

async def transcribe_streaming_parallel(stream, language_code):
    """Transcribe streaming audio with latency measurement."""
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
        model="latest_long",
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=False
    )

    requests = (
        speech.StreamingRecognizeRequest(audio_content=content)
        for content in stream.generator()
    )

    responses = client.streaming_recognize(streaming_config, requests)

    for response in responses:
        for result in response.results:
            if result.is_final:
                start_time = time.time()  # Start latency measurement for transcription
                transcript = result.alternatives[0].transcript
                end_time = time.time()  # End latency measurement for transcription

                transcription_latency = end_time - start_time
                print(f"User said: {transcript}")
                print(f"Transcription latency: {transcription_latency:.2f} seconds")

                gemini_response = get_dialogflow_response(transcript, language_code, agent, session_id, flow_id)
                print(f"Gemini responds: {gemini_response}")

                yield gemini_response  # Yielding response to make this an async generator
@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    """Handle the WebSocket connection for media streaming."""
    await websocket.accept()
    stream_handler = AudioStreamHandler()

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            print(data)

            if data['event'] == "media":
                payload = data['media']['payload']
                chunk = base64.b64decode(payload)
                stream_handler.fill_buffer(chunk)

            # Once enough audio is received, start transcription and generate response
            async for response in transcribe_streaming_parallel(stream_handler, "en-US"):
                await websocket.send_text(response)  # Send the response back to Exotel

    except WebSocketDisconnect:
        print("WebSocket disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)