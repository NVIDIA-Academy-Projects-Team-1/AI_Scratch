function getResponse() {
    $.ajax({
        type: "GET",
        url: "/response",
        dataType: "text",
        success: function(data){
            console.log(data)
            var logDiv = document.getElementById("log");
            var log = document.createTextNode(data);

            logDiv.appendChild(log);
        }
    });
}

