/*
    testHandler: Handle test container
*/


document.addEventListener("DOMContentLoaded", function() {
    /*
        Disable test divs by input type
    */
    const targetNode = document.getElementById("constructModel");
    const config = {childList : true};

    const callback = (mutationList, observer) => {
        for(const mutation of mutationList){
            if(mutation.type === "childList"){
                if(targetNode.querySelector("#input-number") != null){
                    document.getElementById("num-test-page").style.display = "block";
                    document.getElementById("image-test-page").style.display = "none";
                    document.getElementById("text-test-page").style.display = "none";
                }
                else if(targetNode.querySelector("#input-image") != null){
                    document.getElementById("num-test-page").style.display = "none";
                    document.getElementById("image-test-page").style.display = "block";
                    document.getElementById("text-test-page").style.display = "none";
                }
                else if(targetNode.querySelector("#input-text") != null){
                    document.getElementById("num-test-page").style.display = "none";
                    document.getElementById("image-test-page").style.display = "none";
                    document.getElementById("text-test-page").style.display = "block";
                }
            }
        }
    };
    const observer = new MutationObserver(callback);
    observer.observe(targetNode, config);

    /*
        Send server test values and receive prediction response
    */
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
            }
        });
   }
});