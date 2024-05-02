/*
    nodeHandler: Handle node behavior
*/


document.addEventListener("DOMContentLoaded", function() {
    const trainButton = document.getElementById("train-model")

    trainButton.onclick = function() {
   
        var x_val = document.getElementById('x').value;
        var y_val = document.getElementById('y').value;
        var img_val = document.getElementById('img').value;
        var text_val = document.getElementById('text').value;
        var units_val = document.getElementById('units').value;
        var act_val = document.getElementById('activationFunc').value;

        // console.log(x_val);
        // console.log(y_val);
        // console.log(img_val);
        // console.log(text_val);
        // console.log(units_val);
        // console.log(act_val);

    }
});