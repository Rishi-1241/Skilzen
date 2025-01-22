import base64

# Read the MP3 file and convert it to base64
with open("C:\\Users\\Prakhar Agrawal\\Downloads\\audio3.wav", "rb") as audio_file:
    encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')

# Ensure proper padding
# Add padding if needed
print(len(encoded_audio))
padding_needed = len(encoded_audio) % 4
if padding_needed > 0:
    encoded_audio += '=' * (4 - padding_needed)

# Print or send the base64 string
print(encoded_audio)
