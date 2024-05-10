/*
    testHandler: Handle model test div container
    * Send server test data
    * Process response from server
*/


document.addEventListener("DOMContentLoaded", function() {
    // Disable test button when image/text input node has been selected
    const targetNode = document.getElementById("constructModel");
    const config = {childList : true};

    const callback = (mutationList, observer) => {
        for(const mutation of mutationList){
            if(mutation.type === "childList"){
                if(targetNode.querySelector("#input-number") != null){
                    document.getElementById("num-test-page").style.display = "block";
                }
            }
        }
    };
    const observer = new MutationObserver(callback);
    observer.observe(targetNode, config);


   // Send server test values and receive prediction response
   const testNumberButton = document.getElementById("test-num-button");

   testNumberButton.onclick = function() {
        var x_val = document.getElementById('test-x').value;

        if(/[a-z]/.exec(x_val) != null){
            alert("콤마로 구분된 숫자만 입력 가능합니다");
            return;
        }

        var data = {
            x: x_val
        }

        $.ajax({
            url: "/testdata",
            type: "POST",
            data: data,
            success: (res) => {
                console.log(res);

                var logDiv = document.getElementById("num-test-output");
                var lines = res.split('\n');
                
                lines.forEach(line => {
                    var log = document.createTextNode(line);
                    logDiv.appendChild(log);
                    logDiv.appendChild(document.createElement('br'));
                });
                logDiv.append("----------------------------")
            }
        });
   }
});