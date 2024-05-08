/*
    index.js : Handle layer visibility
    * Show selected layer and hide others

    TODO : Disable input layer when hidden layer has been selected
*/

document.addEventListener("DOMContentLoaded", function() {
    document.getElementById("layer1").style.display = "block";

    document.querySelector("#input-layer-button").onclick = function(){
        document.getElementById("layer1").style.display = "block";
        document.getElementById("layer2").style.display = "none";
        document.getElementById("layer3").style.display = "none";
    }

    document.querySelector("#hidden-layer-button").onclick = function(){
        document.getElementById("layer1").style.display = "none";
        document.getElementById("layer2").style.display = "block";
        document.getElementById("layer3").style.display = "none";
    }

    document.querySelector("#run-layer-button").onclick = function(){
        document.getElementById("layer1").style.display = "none";
        document.getElementById("layer2").style.display = "none";
        document.getElementById("layer3").style.display = "block";
    }
    
    document.getElementById("reset").addEventListener("click", function(){
        console.log("reload");
        location.reload()
    });
});