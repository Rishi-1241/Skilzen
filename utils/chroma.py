import io
import pyaudio
import google.generativeai as genai
from google.cloud import speech
from google.cloud import texttospeech
import os
from pydub import AudioSegment
from pydub.playback import play
import time


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../credentials_skilzen.json"
GOOGLE_API_KEY = "AIzaSyCnP8cuR-cfpydBlHYbFv0fAtJhqdpHQKQ"
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

def get_gemini_response(text, language_code):
    """Get response from Gemini model"""
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = f"""
    Respond to the following query in the same language as the query.
    Query: {text}
    Language code: {language_code}
    Please provide a natural and helpful response don't use any special characters.
    """

    start_time = time.time()  # Start latency measurement
    response = model.generate_content(prompt)
    end_time = time.time()  # End latency measurement

    latency = end_time - start_time
    print(f"Gemini response latency: {latency:.2f} seconds")

    return response.text

import os
import tempfile
from openai import OpenAI

client = OpenAI()
os.environ["OPENAI_API_KEY"] = "sk-c-cMt0Ej5AsKuL_rdID66WfxbOO8En9Mk-uhsTBLJCT3BlbkFJFyrLuvPvLUDbhRGZX5Lyrl4nx1LAXnbCECV9bQagEA"

import os
import tempfile
from openai import OpenAI
from pydub import AudioSegment

client = OpenAI()

def transcribe_streaming(stream, language_code):
    """Transcribe streaming audio with latency measurement"""
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
                start_time = time.time()  # Start latency measurement for transcription
                transcript = result.alternatives[0].transcript
                end_time = time.time()  # End latency measurement for transcription

                transcription_latency = end_time - start_time
                print(f"User said: {transcript}")
                print(f"Transcription latency: {transcription_latency:.2f} seconds")

                gemini_response = get_gemini_response(transcript, language_code)
                print(f"Gemini responds: {gemini_response}")

                yield gemini_response


import asyncio

response_audio_cache = {}

def synthesize_text(text, language):
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



def audio_stream_generator():
    """Generate audio stream from microphone"""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=256
    )

    print("Listening... (Press Ctrl+C to stop)")
    try:
        while True:
            data = stream.read(1024, exception_on_overflow=False)
            yield data
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("\nStopped listening.")

def main(language):
    """Main function to run the voice assistant"""
    try:
        if language not in LANGUAGE_CONFIGS:
            raise ValueError(f"Unsupported language: {language}")

        lang_config = LANGUAGE_CONFIGS[language]
        print(f"Starting voice assistant in {language} using {lang_config['voice_name']}...")
        for response_text in transcribe_streaming(audio_stream_generator(), lang_config["code"]):
            audio_content = synthesize_text(response_text, language)
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_content), format="wav")
            play(audio_segment)

    except KeyboardInterrupt:
        print("\nExiting voice assistant...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    language = "Hindi"  
    main(language)
