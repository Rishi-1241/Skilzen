
@socketio.on('audio_stream')
def handle_audio_stream(audio_data):
    """Handle incoming audio chunks from the frontend."""
    print("Received audio chunk")
    language_code = "en-US"  # Replace with your language code

    def process():
        async def async_process():
            print("transcribing")
            async for transcript in transcribe_streaming_parallel([audio_data], language_code):
                # Step 2: Generate response using your model and Dialogflow
                # gemini_response = get_dialogflow_response(transcript, language_code, agent, session_id, flow_id)
                print(f"Gemini responds: {transcript}")

                # Step 3: Convert response text to audio using TTS
                audio_content = await synthesize_text_parallel(transcript, language_code)

                # Step 4: Send the audio file back to the frontend
                emit('audio_response', audio_content)

        # Run the async function in the background
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(async_process())

    # Start the background task
    process()
