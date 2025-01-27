import json
import base64
import queue
import time
from fastapi import FastAPI, WebSocket
from google.cloud import speech
from pydub import AudioSegment
import io
from typing import AsyncGenerator, Generator

app = FastAPI()
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
project_id = "certain-math-447716-d1"  
location_id = "global"  
agent_id = "ab039e5f-d9ce-4feb-90ad-4184f23f01e5"  
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

# Create a buffer to store incoming audio chunks
class AudioStreamBuffer:
    def __init__(self):
        self.buff = queue.Queue()

    def fill_buffer(self, in_data):
        """Continuously collect data from the audio stream, into the buffer."""
        self.buff.put(in_data)

    def generator(self) -> Generator[bytes, None, None]:
        """Generate audio chunks from the buffer."""
        while True:
            chunk = self.buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Consume any other data in the buffer.
            while True:
                try:
                    chunk = self.buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

# Initialize the audio buffer
audio_stream = AudioStreamBuffer()

async def transcribe_streaming_parallel(stream, language_code):
    print("Transcribing streaming audio...")
    """Transcribe streaming audio with latency measurement."""
    client = speech.SpeechClient()
    print("Client created")

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000,
        language_code=language_code,
        model="latest_long",
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=False
    )

    requests = (
        speech.StreamingRecognizeRequest(audio_content=content)
        for content in stream
    )

    responses = client.streaming_recognize(streaming_config, requests)
    print("Responses received")
    print(responses)

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint to receive streaming audio data."""
    await websocket.accept()
    has_seen_media = False
    message_count = 0

    while True:
        message = await websocket.receive_text()  # Receive message from client
        if message is None:
            continue

        # Decode and process the incoming message
        data = json.loads(message)

        if data['event'] == "connected":
            print("Connected Message received")

        if data['event'] == "start":
            print("Start Message received")

        if data['event'] == "media":
            # Extract the payload and decode the audio chunk
            payload = data['media']['payload']
            print(payload)
            chunk = base64.b64decode(payload)

            # Fill the buffer with the incoming audio data
            audio_stream.fill_buffer(chunk)

            # Process the buffered audio for transcription
            audio_generator = audio_stream.generator()
            try:
                # Start streaming and transcribing the collected chunks
                async for response_text in transcribe_streaming_parallel(audio_generator, 'en-US'):
                    print(f"Response: {response_text}")
            except Exception as e:
                print(f"Error during transcription: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
