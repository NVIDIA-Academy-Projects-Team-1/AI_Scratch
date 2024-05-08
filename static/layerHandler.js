/*
    layerHandler.js : Handle event when node block has dragged from layer selection container
    * Hide placeholdertext when node block is placed inside consturctModel div container
*/


document.addEventListener("DOMContentLoaded", function() {
    const targetNode = document.getElementById("constructModel");
    const config = {childList : true};

    const callback = (mutationList, observer) => {
        for(const mutation of mutationList){
            if(mutation.type === "childList"){
                if(targetNode.children.length === 1){
                    document.getElementById("placeholdertext").style.display = "block";
                }
                else{
                    document.getElementById("placeholdertext").style.display = "none";
                }
            }
        }
    };
    const observer = new MutationObserver(callback);
    observer.observe(targetNode, config);


});