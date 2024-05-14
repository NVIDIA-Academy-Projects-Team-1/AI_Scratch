var isRecording = false;
var mediaRecorder;
var recordedChunks = [];

document.addEventListener("DOMContentLoaded", function() {
    document.getElementById('recordButton').addEventListener('click', function() {
        if (isRecording) {
            document.getElementById("recordingIndicator").style.display = "none";
            mediaRecorder.stop();
            isRecording = false;

            var blob = new Blob(recordedChunks, { type: 'audio/wav' });
            var formData = new FormData();
            formData.append('audio', blob);

            // fetch('/audio_to_text', {
            //     method: 'POST',
            //     body: formData
            // })
            // .then(response => response.json())
            // .then(data => {
            //     if (data.error) {
            //         document.getElementById('transcription').textContent = data.error;
            //     } else {
            //         document.getElementById('transcription').textContent = data.text;
            //     }
            // })
            // .catch(error => {
            //     console.error('오류 발생:', error);
            // });

            var xhr = new XMLHttpRequest();
        
            xhr.open('POST', '/audio_to_text', true);
            xhr.onload = function() {
                if (xhr.status === 200) {
                    document.getElementById('transcription').textContent = xhr.response.text;
                } else {
                    document.getElementById('transcription').textContent = xhr.response['error'];
                }
            };
            xhr.send(formData);
    

            recordedChunks = [];
        }
        else {
            document.getElementById("recordingIndicator").style.display = "block";
            isRecording = true;
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(function(stream) {
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.ondataavailable = function(event) {
                        if (event.data.size > 0) {
                            recordedChunks.push(event.data);
                        }
                    };
                    mediaRecorder.start();
                })
                .catch(function(err) {
                    console.log('오류 발생: ' + err);
                });
        }
    });
});
