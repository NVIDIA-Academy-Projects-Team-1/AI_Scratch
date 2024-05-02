/*
    nodeHandler: Handle node behavior
*/


document.addEventListener("DOMContentLoaded", function() {
    const targetNode = document.getElementById("constructModel");
    const config = {childList : true};

    const callback = (mutationList, observer) => {
        for(const mutation of mutationList){
            if(mutation.type === "childList"){
                if(targetNode.querySelector("#train-model") != null){
                    targetNode.querySelector("#train-model").disabled = false;
                }
                else if(targetNode.querySelector("#generate-text") != null){
                    targetNode.querySelector("#generate-text").disabled = false;
                }
                else{
                    document.querySelector("#generate-text").disabled = true;
                    document.querySelector("#train-model").disabled = true;
                }   
            }
        }
    };
    const observer = new MutationObserver(callback);
    observer.observe(targetNode, config);
});