/*
    handleResponse.js : Handle response from server

    TODO : Not Working Properly, all responses are now being processed in trainModel.js/testHandler.js
*/



function getResponse() {
    $.ajax({
        type: "GET",
        url: "/response",
        dataType: "json",
        success: function(data) {
            console.log(data);
            var logDiv = document.getElementById("log");
            var log = document.createTextNode(JSON.stringify(data));

            logDiv.appendChild(log);
        }
    });
}