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

        var class_val = document.getElementById('class').value;
        var x_log_val = document.getElementById('x-log').value;
        var y_log_val = document.getElementById('y-log').value;

        var units_val_1 = document.getElementById('units1').value;
        var units_val_2 = document.getElementById('units2').value;
        var units_val_3 = document.getElementById('units3').value;

        var act_val_1 = document.getElementById('activationFunc1').value;
        var act_val_2 = document.getElementById('activationFunc2').value;
        var act_val_3 = document.getElementById('activationFunc3').value;

        var parent = document.getElementById("constructModel");
        
        // Exception handling
        if(units_val_1 != '' && (units_val_1 <= 0 || units_val_1 > 128)){
            alert("뉴런의 개수는 음수이거나 128개를 넘을 수 없습니다.");
            return;
        }
        else if(units_val_2 != '' && (units_val_2 <= 0 || units_val_2 > 128)){
            alert("뉴런의 개수는 음수이거나 128개를 넘을 수 없습니다.");
            return;
        }
        else if(units_val_3 != '' && (units_val_3 <= 0 || units_val_3 > 128)){
            alert("뉴런의 개수는 음수이거나 128개를 넘을 수 없습니다.");
            return;
        }

        if(parent.querySelector("#input-number")){
            if(/[a-z]/.exec(x_val) != null || /[a-z]/.exec(y_val) != null){
                alert("콤마로 구분된 숫자만 입력 가능합니다");
                return;
            }
            if(x_val.split(',').length != y_val.split(',').length){
                alert("시작값과 목표값의 개수가 다릅니다.");
                return;
            }
            if(x_val === '' || y_val === ''){
                alert("시작값과 목표값을 입력해 주세요");
                return;
            }
        }
        else if(parent.querySelector("#input-number-logistic")){
            var y_log_val_list = y_log_val.split(',').map(Number);

            if(/[a-z]/.exec(x_log_val) != null || /[a-z]/.exec(y_log_val) != null || /[a-z]/.exec(class_val) != null){
                alert("콤마로 구분된 숫자만 입력 가능합니다");
                return;
            }
            else if(class_val <= 0 || class_val > 9){
                alert("클래스 개수는 0보다 크고 10보다 작아야 합니다.");
                return;
            }
            else if(y_log_val_list.includes(0) || y_log_val_list.some(x => x > class_val)){
                alert("클래스 번호는 1부터 클래스 개수까지여야 합니다.");
                return;
            }

            if(x_log_val.split(',').length != y_log_val.split(',').length){
                alert("조건값과 클래스값의 개수가 다릅니다.")
                return;
            }
            if(x_log_val === '' || y_log_val === ''){
                alert("조건값과 클래스값을 입력해 주세요.");
                return;
            }
        }

        if(parent.querySelector("#input-image") || parent.querySelector("#input-text")){
            alert("모델 훈련하기는 숫자 예측에만 사용할 수 있습니다");
            return;
        }

        // Data aggregation and server requesting
        var data = {
            type: "number",
            reg_type: class_val != '' ? "logistic" : "linear",
            class: class_val,
            x: x_val,
            y: y_val,
            x_log: x_log_val,
            y_log: y_log_val,
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

        if(parent.querySelector("#input-number") || parent.querySelector("#input-number-logistic")){
            alert("모델에 질문하기는 이미지와 챗봇에만 사용할 수 있습니다");
            return;
        }

        var formData = new FormData();

        if(parent.querySelector("#input-image") != null){
            if(img_val == null){
                alert('선택된 이미지가 없습니다. 이미지를 선택해주세요.');
            } else{
                formData.append('type','image');
                formData.append('img', img_val);
            }
        }
        else{
            formData.append('type','text');
            formData.append('text',text_val);
        }

        // when image is not existed use only text response, if they have use image with text response
        if(parent.querySelector('#input-image') == null){
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
        else{
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
                    var img = document.createElement('img');
                    img.src = res.image_path;
                    img.style.width = '50%';
                    img.style.height = 'auto';
                    logDiv.innerHTML = '';
                    logDiv.appendChild(img);
                    logDiv.appendChild(responseText);
                }
            });
        }
    }
});