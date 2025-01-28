console.log('script.js loaded');

const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const transcriptDiv = document.getElementById('transcript');
const transcriptDiv2 = document.getElementById('transcript2');
const clear = document.getElementById('stop-btn2');
let mediaRecorder;
let socket;

startBtn.addEventListener('click', async () => {
  startBtn.disabled = true;
  stopBtn.disabled = false;

  // Initialize WebSocket connection
  socket = io(); // Connect to the server
  socket.on('connect', () => {
    console.log('Connected to the server');
  });

  socket.on('connect_error', (error) => {
    console.error('Connection error:', error);
  });
  socket.on("text-transcribe",(data)=>{
    console.log("display")
    transcriptDiv2.innerText += data + '\n';
})
  socket.on('transcription', (audioBuffer) => {
    console.log('Received audio',audioBuffer);
    // Ensure that audioBuffer is valid before proceeding
   
  
    if (audioBuffer && audioBuffer instanceof ArrayBuffer) {
        const audioBlob = new Blob([audioBuffer], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audio.play();
    
        // Optionally, you can display an audio player for user control
        transcriptDiv.innerHTML = `<audio controls autoplay>
                                    <source src="${audioUrl}" type="audio/wav">
                                  </audio>`;
      } else {
        console.error('Received invalid audio data');
      }
  });

  // Capture audio from the microphone
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm; codecs=opus' });

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        // Emit audio chunk as Blob to server
        socket.emit('audioChunk', event.data); // Send audio chunks to the server
      }
    };

    mediaRecorder.start(500); // Send chunks every 500ms
    console.log('MediaRecorder started');
  } catch (error) {
    console.error('Error accessing microphone:', error);
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }
});

stopBtn.addEventListener('click', () => {
  startBtn.disabled = false;
  stopBtn.disabled = true;

  if (mediaRecorder) {
    mediaRecorder.stop();
    console.log('MediaRecorder stopped');
  }

});
clear.addEventListener('click',()=>{
    if (socket) {
        socket.disconnect();
        console.log('Disconnected from the server');
    }
})