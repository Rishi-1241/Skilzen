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
        for content in stream
    )
    

    responses = client.streaming_recognize(streaming_config, requests)

    for response in responses:
        for result in response.results:
            if result.is_final:
                start_time = time.time()  
               # Start latency measurement for transcription
                transcript = result.alternatives[0].transcript
                end_time = time.time()  # End latency measurement for transcription

                transcription_latency = end_time - start_time
                print(f"User said: {transcript}")
                print(f"Transcription latency: {transcription_latency:.2f} seconds")

                gemini_response = get_dialogflow_response(transcript, language_code,agent, session_id, flow_id)
                print(f"Gemini responds: {gemini_response}")

                yield gemini_response  # Yielding response to make this an async generator




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



def audio_stream_generator(timeout=2):
    """Generate audio stream from microphone and stop when no speech is detected for a certain timeout"""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=256
    )

    print("Listening... (Press Ctrl+C to stop)")
    last_activity_time = time.time()

    try:
        while True:
            data = stream.read(1024, exception_on_overflow=False)
            yield data

            # Check if there is silence (i.e., no activity in the stream for a while)
            if max(abs(int(i)) for i in data) < 10:  # Threshold for silence (you can adjust this value)
                if time.time() - last_activity_time > timeout:
                    print("No speech detected for {} seconds, stopping...".format(timeout))
                    break
            else:
                last_activity_time = time.time()

    except KeyboardInterrupt:
        pass

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("\nAudio listened.")



async def main(language):
    """Main function to run the voice assistant"""
    try:
        if language not in LANGUAGE_CONFIGS:
            raise ValueError(f"Unsupported language: {language}")

        lang_config = LANGUAGE_CONFIGS[language]
        print(f"Starting voice assistant in {language} using {lang_config['voice_name']}...")
        async for response_text in transcribe_streaming_parallel(audio_stream_generator(), lang_config["code"]):
            audio_content = synthesize_text_parallel(response_text, language)
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_content), format="wav")
            play(audio_segment)

    except KeyboardInterrupt:
        print("\nExiting voice assistant...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    language = "English"  
    asyncio.run(main(language))
