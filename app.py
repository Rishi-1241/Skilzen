from flask import Flask, render_template, request, jsonify
from google.cloud import speech, texttospeech
from pydub import AudioSegment
from pydub.playback import play
import io
import os
import pyaudio
import wave
import threading
import queue
from dotenv import load_dotenv
from flask_cors import CORS
from google.cloud.dialogflowcx_v3 import SessionsClient
from google.cloud.dialogflowcx_v3.types import session
import uuid

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Update the path to the new credentials file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "dialogflow_credentials.json"
CORS(app)
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

# Constants
project_id = "certain-math-447716-d1"  # Your Google Cloud project ID
location_id = "global"  # Your agent's location, e.g., "global"
agent_id = "ab039e5f-d9ce-4feb-90ad-4184f23f01e5"  # Your Dialogflow CX agent ID
flow_id = "dd90ab06-761a-410d-bb04-f60368c323ac"
agent = f"projects/{project_id}/locations/{location_id}/agents/{agent_id}"

# Language Configurations
LANGUAGE_CONFIGS = {
    "English": {
        "code": "en-US",
        "voice_name": "en-US-Wavenet-F",
        "fallback_voice": "en-US-Standard-C"
    },
    "Telugu": {
        "code": "te-IN",
        "voice_name": "te-IN-Standard-A",
        "fallback_voice": "te-IN-Standard-A"
    },
    "Hindi": {
        "code": "hi-IN",
        "voice_name": "hi-IN-Wavenet-D",
        "fallback_voice": "hi-IN-Standard-D"
    },
    "Tamil": {
        "code": "ta-IN",
        "voice_name": "ta-IN-Standard-A",
        "fallback_voice": "ta-IN-Standard-A"
    },
    "Kannada": {
        "code": "kn-IN",
        "voice_name": "kn-IN-Standard-A",
        "fallback_voice": "kn-IN-Standard-A"
    },
    "Malayalam": {
        "code": "ml-IN",
        "voice_name": "ml-IN-Standard-A",
        "fallback_voice": "ml-IN-Standard-A"
    }
}

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5  # Default recording duration

class AudioRecorder:
    def __init__(self):
        try:
            self.audio = pyaudio.PyAudio()
        except Exception as e:
            print(f"Error initializing PyAudio: {str(e)}")
            self.audio = None
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.audio_queue = queue.Queue()

    def start_recording(self):
        if self.audio is None:
            print("PyAudio is not initialized. Cannot start recording.")
            return

        self.frames = []
        self.is_recording = True
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        def record():
            while self.is_recording:
                data = self.stream.read(CHUNK)
                self.frames.append(data)
                self.audio_queue.put(data)

        self.recording_thread = threading.Thread(target=record)
        self.recording_thread.start()

    def stop_recording(self):
        if self.audio is None:
            print("PyAudio is not initialized. Cannot stop recording.")
            return

        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        # Save recording to WAV file
        with wave.open("temp_recording.wav", 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))

        return "temp_recording.wav"

    def __del__(self):
        if self.audio is not None:
            self.audio.terminate()

def get_dialogflow_response(text, language_code):
    """Get response from Dialogflow CX agent"""
    session_id = str(uuid.uuid4())
    environment_id = "draft"
    session_path = f"{agent}/environments/{environment_id}/sessions/{session_id}?flow={flow_id}"

    text_input = session.TextInput(text=text)
    query_input = session.QueryInput(text=text_input, language_code=language_code)
    request_obj = session.DetectIntentRequest(
        session=session_path,
        query_input=query_input,
    )

    session_client = SessionsClient()
    response = session_client.detect_intent(request=request_obj)

    response_messages = [
        " ".join(msg.text.text) for msg in response.query_result.response_messages
    ]
    return " ".join(response_messages)

def transcribe_audio_file(file_path, language_code):
    """Transcribe audio file"""
    client = speech.SpeechClient()

    with open(file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        return result.alternatives[0].transcript

def synthesize_text(text, language):
    """Synthesize text to speech using WaveNet voices"""
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    lang_config = LANGUAGE_CONFIGS[language]
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
    try:
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
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

# Create global audio recorder instance
audio_recorder = AudioRecorder()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start-recording', methods=['POST'])
def start_recording():
    audio_recorder.start_recording()
    return jsonify({"status": "Recording started"})

@app.route('/stop-recording', methods=['POST'])
def stop_recording():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid request data"}), 400

    language = data.get('language', 'English')
    audio_file = audio_recorder.stop_recording()
    transcript = transcribe_audio_file(audio_file, LANGUAGE_CONFIGS[language]["code"])
    if transcript:
        dialogflow_response = get_dialogflow_response(transcript, LANGUAGE_CONFIGS[language]["code"])
        # Return the text response immediately
        return jsonify({
            "status": "success",
            "transcript": transcript,
            "response": dialogflow_response
        })
    return jsonify({"status": "error", "message": "No speech detected"})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    transcriptions = []
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid request data"}), 400

    language = data.get('language', 'English')
    if 'text' in data:
        user_input = data['text']
        dialogflow_response = get_dialogflow_response(user_input, LANGUAGE_CONFIGS[language]["code"])
        transcriptions.append({"user": user_input, "bot": dialogflow_response})
        # Return the text response immediately
        return jsonify({"transcriptions": transcriptions})

@app.route('/speak', methods=['POST'])
def speak():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid request data"}), 400

    # Debugging statements
    print(f"Received data: {data}")

    text = data.get('text')
    language = data.get('language')

    if not text or not language:
        return jsonify({"error": "Missing 'text' or 'language' parameter"}), 400

    audio_content = synthesize_text(text, language)
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_content), format="wav")
    play(audio_segment)
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run()
