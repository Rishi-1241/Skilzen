import io
import re
import sys
import argparse
import time

from google.cloud import speech_v1p1beta1 as speech
import base64
import json
import signal
import logging
import threading


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
import base64
import io
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketState
import uvicorn

app = FastAPI()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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



import queue
import time
import threading
from google.cloud import speech

class Stream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self.buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self.buff.put(None)

    def fill_buffer(self, in_data):
        """Continuously collect data from the audio stream, into the buffer."""
        self.buff.put(in_data)
        return self

    def generator(self):
        while True:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.

            print("generator function called")
            chunk = self.buff.get()
            if chunk is None:
                print("chunk is none")
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self.buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

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



def listen_print_loop(responses):
    for response in responses:
        for result in response.results:
            if result.is_final:
                start_time = time.time()  # Start latency measurement
                transcript = result.alternatives[0].transcript
                end_time = time.time()  # End latency measurement

                transcription_latency = end_time - start_time
                print(f"User said: {transcript}")
                print(f"Transcription latency: {transcription_latency:.2f} seconds")

                # Process with Dialogflow
                gemini_response = get_dialogflow_response(transcript, language_code, agent, session_id, flow_id)
                print(f"Gemini responds: {gemini_response}")

                yield gemini_response

def stream_transcript(stream,client,config):
    audio_generator = stream.generator()
    print("inside streaming transcript")
    while True:
       
        try:
            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )
            responses = client.streaming_recognize(config, requests)
            yield from listen_print_loop(responses)
        except Exception as e:
            print(f"Error during streaming recognition: {str(e)}")
        time.sleep(5)
is_websocket_closed = False
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global is_websocket_closed
    await websocket.accept()
    print("WebSocket connection accepted")

    stream = Stream(RATE, CHUNK)

    client = speech.SpeechClient()
    print("Client created")

    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,
        max_speaker_count=10,
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000,
        language_code="en-US",
        diarization_config=diarization_config,
    )

    transcription_thread = threading.Thread(
        target=stream_transcript,
        args=(stream,client,config), 
        daemon=True
    )
    transcription_thread.start()
    print("thread started")

    try:
        while True:
            data = await websocket.receive_text()
            if data == "disconnect":
                print("WebSocket connection closed by client")
                break

            if data:
                data = json.loads(data)
                print(data)

            if data['event'] == "media":
                #print("got the media payload")
                payload = data['media']['payload']
                print(payload)
                chunk = base64.b64decode(payload)
                stream.fill_buffer(chunk)
            
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        stream.buff.put(None)
        if not is_websocket_closed:
            is_websocket_closed = True
            await websocket.close()


# Parameters
RATE = 8000
CHUNK = int(RATE / 10)  # 100ms audio chunks
language_code = "en-IN"

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)   