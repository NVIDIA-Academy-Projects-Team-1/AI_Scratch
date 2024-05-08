/*
    trainModel.js : Handle train/query action
    * Send server number/image/text data
    * Process response from server
*/


document.addEventListener("DOMContentLoaded", function() {
    const trainButton = document.getElementById("train-model")
    const queryButton = document.getElementById("query-model")

    // When input is number
    trainButton.onclick = function() {
        document.querySelector("#test-num-button").disabled = false;
   
        var x_val = document.getElementById('x').value;
        var y_val = document.getElementById('y').value;
        // var img_val = document.getElementById('img').value;
        var units_val_1 = document.getElementById('units1').value;
        var units_val_2 = document.getElementById('units2').value;
        var units_val_3 = document.getElementById('units3').value;
        var act_val_1 = document.getElementById('activationFunc1').value;
        var act_val_2 = document.getElementById('activationFunc2').value;
        var act_val_3 = document.getElementById('activationFunc3').value;

        var parent = document.getElementById("constructModel");

        if(parent.querySelector("#input-number")){
            if(/[a-z]/.exec(x_val) != null || /[a-z]/.exec(y_val) != null){
                alert("콤마로 구분된 숫자만 입력 가능합니다");
                return;
            }
        }
        else{
            alert("모델 훈련은 숫자 예측에만 사용할 수 있습니다");
            return;
        }

        var data = {
            type: "number",
            x: x_val,
            y: y_val,
            // img: img_val,
            units1: units_val_1,
            units2: units_val_2,
            units3: units_val_3,
            activation1: act_val_1,
            activation2: act_val_2,
            activation3: act_val_3,
        }

        $.ajax({
            url: "/data",
            type: "POST",
            data: data,
            success: (res) => {
                console.log(res);

                var logDiv = document.getElementById("log");
                var lines = res.split('\n');
                
                lines.forEach(line => {
                    var log = document.createTextNode(line);
                    logDiv.appendChild(log);
                    logDiv.appendChild(document.createElement('br'));
                });

                logDiv.appendChild(document.createTextNode("학습이 완료되었습니다."));
                logDiv.appendChild(document.createElement('br'));
            }
        });
    }

    
    // When input is image/text
    queryButton.onclick = function() {
        var text_val = document.getElementById('text').value;
        var img_val = document.getElementById('img').files[0];
        var parent = document.getElementById("constructModel");
        var type;

        if(parent.querySelector("#input-image") === null && text_val === ''){
            alert("모델에게 질문하기는 이미지 또는 채팅에만 사용할 수 있습니다")
            return;
        }

        var formData = new FormData();

        if(parent.querySelector("#input-image") != null){
            formData.append('type','image');
            formData.append('img', img_val);
        }
        else{
            formData.append('type','text');
            formData.append('text',text_val);
        }
        var fileDOM = document.querySelector('#file');
        var preview = document.querySelector('.image-box');

        fileDOM.addEventListener('change',() => { 
            var imageSrc 
        })
        $.ajax({
            url: "/data",
            type: "POST",
            processData: false,
            contentType: false,
            data: formData,
            success: (res) => {
                console.log(res);
                var logDiv = document.getElementById("log");
                var responseText = document.createTextNode(res.response);
                logDiv.innerHTML = '';
                logDiv.appendChild(responseText);
            }
        });
        
    }

});