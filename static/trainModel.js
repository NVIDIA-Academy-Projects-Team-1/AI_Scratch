/*
    trainModel.js : Handle train/query action
    * Send server number/image/text data
    * Process response from server
*/


document.addEventListener("DOMContentLoaded", function() {
    const trainButton = document.getElementById("train-model")
    const queryButton = document.getElementById("query-model")

    // Handle number inputs
    trainButton.onclick = function() {
        document.querySelector("#test-num-button").disabled = false;
   
        var x_val = document.getElementById('x').value;
        var y_val = document.getElementById('y').value;

        var x_file_val = document.getElementById('x-file').files[0];
        var y_file_val = document.getElementById('y-file').files[0];

        var csv_file_val = document.getElementById('csv-file').files[0];

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
        
        /* Exception handling */
        // Dense layer exception handling
        if(parent.querySelector("#units1") == null && parent.querySelector("#units2") == null && parent.querySelector("#units3") == null){
            alert("모델 훈련시에는 최소한 하나의 뉴런이 필요합니다.");
            return;
        }
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
        // Inputs exception handling
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
        else if(parent.querySelector("#input-number-file")){
            if(x_file_val == null || y_file_val == null){
                alert("X값 또는 Y값 파일이 올바르게 업로드 되지 않았습니다.");
                return;
            }
        }
        else if(parent.querySelector("#input-csv-file")){
            if(csv_file_val == null){
                alert("CSV 파일이 올바르게 업로드 되지 않았습니다.");
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
        // Invalid input exception handling
        if(parent.querySelector("#input-image") || parent.querySelector("#input-text")){
            alert("모델 훈련하기는 숫자 예측에만 사용할 수 있습니다");
            return;
        }


        /* Data aggregation and server requesting */
        // When input is number(linear or logistic)
        if(parent.querySelector("#input-number") || parent.querySelector("#input-number-logistic")){
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
                        if(line.startsWith('plot')){
                            console.log("plot image called")
                            logDiv.appendChild(document.createTextNode("== 학습 루프별 차이값 그래프 =="));
                            logDiv.appendChild(document.createElement('br'));
                            var img = document.createElement('img');
                            img.src = 'data:image/jpeg;base64,' + line.split(':')[1];
                            img.style.width = '100%';
                            img.style.height = 'auto';
                            logDiv.appendChild(img);
                            logDiv.appendChild(document.createElement('br'));
                        }
                        else{
                            var log = document.createTextNode(line);
                            logDiv.appendChild(log);
                            logDiv.appendChild(document.createElement('br'));
                        }
                        
                    });
    
                    logDiv.appendChild(document.createTextNode("학습이 완료되었습니다."));
                    logDiv.appendChild(document.createElement('br'));
                }
            });
        }
        // When input is number txt file
        else if(parent.querySelector("#input-number-file")){
            var formData = new FormData();
            formData.append('type', "number-file");
            formData.append('reg_type', "linear");
            formData.append('x_file', x_file_val);
            formData.append('y_file', y_file_val);
            formData.append('units1', units_val_1);
            formData.append('units2', units_val_2);
            formData.append('units3', units_val_3);
            formData.append('activation1', act_val_1);
            formData.append('activation2', act_val_2);
            formData.append('activation3', act_val_3);

            $.ajax({
                url: "/data",
                type: "POST",
                data: formData,
                processData: false,
                contentType: false,
                success: (res) => {
                    console.log(res);

                    if(res['alert']){
                        alert(res['alert']);
                        return;
                    }
    
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
        // When input is CSV file
        else if(parent.querySelector("#input-csv-file")){
            var formData = new FormData();
            formData.append('type', "csv-file");
            formData.append('csv_file', csv_file_val);
            formData.append('units1', units_val_1);
            formData.append('units2', units_val_2);
            formData.append('units3', units_val_3);
            formData.append('activation1', act_val_1);
            formData.append('activation2', act_val_2);
            formData.append('activation3', act_val_3);

            $.ajax({
                url: "/data",
                type: "POST",
                data: formData,
                processData: false,
                contentType: false,
                success: (res) => {
                    console.log(csv_file_val['name'])
                    console.log(res);

                    if(res['alert']){
                        alert(res['alert']);
                        return;
                    }
    
                    var logDiv = document.getElementById("log");
                    var lines = res.split('\n');
                    logDiv.appendChild(document.createTextNode("==" + csv_file_val['name']+ " 파일 학습중입니다.=="));
                    logDiv.appendChild(document.createElement('br'));
                    lines.forEach(line => {
                        if(line.startsWith('plot')){
                            console.log("plot image called")
                            logDiv.appendChild(document.createTextNode("== 학습 루프별 차이값/정확도 그래프 =="));
                            logDiv.appendChild(document.createElement('br'));
                            var img = document.createElement('img');
                            img.src = 'data:image/jpeg;base64,' + line.split(':')[1];
                            img.style.width = '100%';
                            img.style.height = 'auto';
                            logDiv.appendChild(img);
                            logDiv.appendChild(document.createElement('br'));
                        }
                        else{
                            var log = document.createTextNode(line);
                            logDiv.appendChild(log);
                            logDiv.appendChild(document.createElement('br'));
                        }

                    });
    
                    logDiv.appendChild(document.createTextNode("학습이 완료되었습니다."));
                    logDiv.appendChild(document.createElement('br'));
                    logDiv.appendChild(document.createElement('br'));
                }
            });
        }
    }

    
    // Handle image or text input
    queryButton.onclick = function() {
        var text_val = document.getElementById('text').value;
        var img_val = document.getElementById('img').files[0];
        var parent = document.getElementById("constructModel");
        var type;

        if(parent.querySelector("#input-number") || parent.querySelector("#input-number-logistic")){
            alert("모델에 질문하기는 이미지와 챗봇에만 사용할 수 있습니다");
            return;
        }

        if(parent.querySelector("#input-audio")){
            text_val = document.getElementById("transcription").innerHTML;
            if(text_val == "응답을 받아오는 중 오류가 발생했습니다."){
                alert("음성이 제대로 인식되지 않았습니다.");
                return;
            }
        }

        var formData = new FormData();

        if(parent.querySelector("#input-image") != null){
            if(img_val == null){
                alert('선택된 이미지가 없습니다. 이미지를 선택해주세요.');
            }
            else{
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
                    var userText = document.createTextNode("사용자: " + text_val);
                    logDiv.appendChild(userText);
                    logDiv.appendChild(document.createElement('br'));

                    var responseText = document.createTextNode("chat: " + res.response);
                    logDiv.appendChild(responseText);
                    logDiv.appendChild(document.createElement('br'));
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
                    var userText = document.createTextNode("사용자: " + text_val);
                    logDiv.appendChild(userText);
                    logDiv.appendChild(document.createElement('br'));

                    var responseText = document.createTextNode("Chat: " + res.response);
                    logDiv.appendChild(responseText);
                    logDiv.appendChild(document.createElement('br'));

                    var img = document.createElement('img');
                    img.src = res.image_path;
                    img.style.width = '50%';
                    img.style.height = 'auto';
                    logDiv.innerHTML = '';
                    logDiv.appendChild(img);
                    logDiv.appendChild(document.createElement('br'));
                    logDiv.appendChild(responseText);
                }
            });
        }
    }
});