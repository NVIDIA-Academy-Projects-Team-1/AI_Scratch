var isRecording = false;
var mediaRecorder;
var recordedChunks = [];

document.addEventListener('DOMContentLoaded', function() {
    const modelLog = document.getElementById('model-log');
    const recordButton = document.getElementById('recordButton');
    const recordingIndicator = document.getElementById('recordingIndicator');
    const transcription = document.getElementById('transcription');
    const queryModelButton = document.getElementById('query-model-button');
    const modelResponseDiv = document.getElementById('model-response');

    recordButton.addEventListener('click', function() {
        if (isRecording) {
            recordingIndicator.style.display = 'none';
            mediaRecorder.stop();
            isRecording = false;
        } else {
            recordingIndicator.style.display = 'block';
            isRecording = true;
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(function(stream) {
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.ondataavailable = function(event) {
                        if (event.data.size > 0) {
                            recordedChunks.push(event.data);
                        }
                    };
                    mediaRecorder.onstop = function() {
                        var blob = new Blob(recordedChunks, { type: 'audio/wav' });
                        var formData = new FormData();
                        formData.append('audio', blob);

                        fetch('/audio_to_text', {
                            method: 'POST',
                            body: formData
                        })
                        .then(response => {
                            if (!response.ok) {
                                throw new Error('Network response was not ok');
                            }
                            return response.json();
                        })
                        .then(data => {
                            transcription.textContent = data.text;
                        })
                        .catch(error => {
                            console.error('Error:', error);
                            transcription.textContent = '응답을 받아오는 중 오류가 발생했습니다.';
                        });

                        recordedChunks = [];
                    };
                    mediaRecorder.start();
                })
                .catch(function(err) {
                    console.log('오류 발생: ' + err);
                });
        }
    });
    
});