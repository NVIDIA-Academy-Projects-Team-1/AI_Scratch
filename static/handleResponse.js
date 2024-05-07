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