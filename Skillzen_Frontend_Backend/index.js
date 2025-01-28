// Import dependencies
const express = require('express');
require('dotenv').config();
const http = require('http');
const { Server } = require('socket.io');
const speech = require('@google-cloud/speech');
const axios = require("axios");
const socketClient = require('socket.io-client');  
// Initialize Express and server
const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*", // Allow all origins or replace with your domain
    methods: ["GET", "POST"]
  }
});
const externalSocket = socketClient('https://skilzen-1088440979862.asia-south1.run.app');
// Google Cloud STT client
const client = new speech.SpeechClient();

// Default STT request configuration
const defaultRequestConfig = {
  encoding: 'WEBM_OPUS',
  sampleRateHertz: 16000,
  languageCode: 'en-US',
  interimResults: true, // Get interim results for real-time transcription
};

// Handle WebSocket connections
io.on('connection', (socket) => {
  console.log('Client connected');

  let recognizeStream = null;

  const startTranscriptionStream = () => {
    console.log('Starting transcription stream...');
    
    recognizeStream = client
      .streamingRecognize({ config: defaultRequestConfig })
      .on('data', async (data) => {
        const transcript = data.results[0]?.alternatives[0]?.transcript || '';
        console.log('Transcription:', transcript);
        socket.emit("text-transcribe",transcript);
        try {
          // Make a POST request to the external API
          
          externalSocket.emit('transcribe', {text:transcript});
          
       console.log("Success")
        } catch (error) {
          // Handle errors gracefully
          console.error('Error while sending transcription to API:', error.response?.data || error.message);
        }
  
        // Emit the transcript back to the connected client
        
      })
      .on('error', (error) => {
        console.error('Error in transcription:', error);
        socket.emit('transcriptionError', error.message);
        stopTranscriptionStream();
      })
      .on('end', () => {
        console.log('Transcription ended');
      });
  };
  
 

  // Stop the transcription stream
  const stopTranscriptionStream = () => {
    if (recognizeStream) {
      console.log('Stopping transcription stream...');
      recognizeStream.end();
      recognizeStream = null;
    }
  };
  externalSocket.on('audio_response', (audioData) => {
    console.log("Got audio file",(audioData))
    socket.emit('transcription', audioData);
    console.log("Audio file sent")
  });
  // Handle audio chunks from the client
  socket.on('audioChunk', (chunk) => {
    if (!recognizeStream) {
      startTranscriptionStream();
    }

    if (recognizeStream) {
      try {
        recognizeStream.write(chunk); // Send audio chunk to Google STT
      } catch (error) {
        console.error('Error writing chunk:', error.message);
      }
    }
  });

  // Handle stop event
  socket.on('stop', () => {
    stopTranscriptionStream();
  });

  // Handle client disconnect
  socket.on('disconnect', () => {
    console.log('Client disconnected');
    stopTranscriptionStream();
  });
});

// Serve static files
app.use(express.static('public'));

// Start the server
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});