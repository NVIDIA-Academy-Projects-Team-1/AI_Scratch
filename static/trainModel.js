/*
    nodeHandler: Handle node behavior
*/


document.addEventListener("DOMContentLoaded", function() {
    const trainButton = document.getElementById("train-model")
    const queryButton = document.getElementById("query-model")

    trainButton.onclick = function() {
   
        var x_val = document.getElementById('x').value;
        var y_val = document.getElementById('y').value;
        // var img_val = document.getElementById('img').value;
        var units_val = document.getElementById('units').value;
        var act_val = document.getElementById('activationFunc').value;

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
            units: units_val,
            activation: act_val
        }

        $.ajax({
            url: "/data",
            type: "POST",
            data: data,
            success: (res) => {
                console.log(res);
            }
        });
    }

    queryButton.onclick = function() {
        var text_val = document.getElementById('img').value;
        var img_val = document.getElementById('img').value;
        
        var parent = document.getElementById("constructModel");
        var type;

        if(parent.querySelector("#input-image") === null){
            alert("모델에게 질문하기는 이미지 또는 채팅에만 사용할 수 있습니다")
            return;
        }
        
        if(parent.querySelector("#input-image") != null){
            type = "image";
        }
        else{
            type = "text";
        }

        var data = {
            type: type,
            img: img_val,
            text: text_val,
        }

        $.ajax({
            url: "/data",
            type: "POST",
            data: data,
            success: (res) => {
                console.log(res);
            }
        });
    }

});