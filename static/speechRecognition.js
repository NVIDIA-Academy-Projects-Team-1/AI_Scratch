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
    //
    function logInteraction(question, answer) {
        const logEntry = document.createElement('div');
        logEntry.classList.add('interaction-entry');
        logEntry.innerHTML = `<strong>질문:</strong> ${question}<br><strong>답변:</strong> ${answer}`;
        modelLog.appendChild(logEntry);
        modelLog.scrollTop = modelLog.scrollHeight;
    }

    function logConversation(question, answer) {
        const logEntry = document.createElement('div');
        logEntry.classList.add('conversation-entry');
        logEntry.innerHTML = `<strong>사용자:</strong> ${question}<br><strong>ollama:</strong> ${answer}`;
        conversationLog.appendChild(logEntry);
        conversationLog.scrollTop = conversationLog.scrollHeight;
    }
    //
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
//
    fetch('/response')
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        logInteraction(data.question, data.answer);
    })
    .catch(error => {
        console.error('Error:', error);
    });
//
});







// var isRecording = false;
// var mediaRecorder;
// var recordedChunks = [];

// let recognizedText = '';

// document.addEventListener('DOMContentLoaded', function() {
//     const recordButton = document.getElementById('recordButton');
//     const recordingIndicator = document.getElementById('recordingIndicator');
//     const transcription = document.getElementById('transcription');
//     const queryModelButton = document.getElementById('query-model-button');
//     const modelResponseDiv = document.getElementById('model-response');

//     let recognition;

//     if (!('webkitSpeechRecognition' in window)) {
//         alert('Your browser does not support speech recognition. Please use Google Chrome.');
//     } else {
//         recognition = new webkitSpeechRecognition();
//         recognition.continuous = false;
//         recognition.interimResults = false;
//         recognition.lang = 'ko-KR';

//         recognition.onstart = function() {
//             recordingIndicator.style.display = 'block';
//         };

//         recognition.onresult = function(event) {
//             recognizedText = event.results[0][0].transcript;
//             transcription.textContent = recognizedText;
//             queryModelButton.disabled = false;
//         };

//         recognition.onerror = function(event) {
//             console.error(event.error);
//         };

//         recognition.onend = function() {
//             recordingIndicator.style.display = 'none';
//         };
//     }

//     recordButton.disabled = false;
//     recordButton.addEventListener('click', function() {
//         if (recognition) {
//             if (recordingIndicator.style.display === 'none') {
//                 recognition.start();
//             } else {
//                 recognition.stop();
//             }
//         }
//     });

//     queryModelButton.addEventListener('click', async function() {
//         const userQuery = recognizedText;

//         try {
//             const response = await fetch('/data', { // 서버 엔드포인트 수정
//                 method: 'POST',
//                 headers: {
//                     'Content-Type': 'application/json'
//                 },
//                 body: JSON.stringify({ type: 'text', text: userQuery }) // JSON 데이터 형식 수정
//             });

//             if (!response.ok) {
//                 throw new Error('Network response was not ok');
//             }

//             const data = await response.json();
//             modelResponseDiv.textContent = data.answer; // 서버 응답 데이터 필드
//         } catch (error) {
//             console.error('Error:', error);
//             modelResponseDiv.textContent = 'Error fetching response from the model.';
//         }
//     });
// });







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