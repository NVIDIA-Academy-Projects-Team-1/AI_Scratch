
// Show node selection container
document.addEventListener("DOMContentLoaded", function() {
    document.getElementById("layer-1").style.display = "block"

    document.querySelector("#input-layer-button").onclick = function(){
        document.getElementById("layer-1").style.display = "block"
        document.getElementById("layer-2").style.display = "none"
        document.getElementById("layer-3").style.display = "none"
    }

    document.querySelector("#hidden-layer-button").onclick = function(){
        document.getElementById("layer-1").style.display = "none"
        document.getElementById("layer-2").style.display = "block"
        document.getElementById("layer-3").style.display = "none"
    }

    document.querySelector("#run-layer-button").onclick = function(){
        document.getElementById("layer-1").style.display = "none"
        document.getElementById("layer-2").style.display = "none"
        document.getElementById("layer-3").style.display = "block"
    }

    document.getElementById("jb-title").addEventListener("mouseover", function(){
        document.getElementById("jb-text").style.display = "block"
    });
    document.getElementById("jb-title").addEventListener("mouseout", function(){
        document.getElementById("jb-text").style.display = "none"
    });

    
});