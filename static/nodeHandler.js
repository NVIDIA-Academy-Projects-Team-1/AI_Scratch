/*
    nodeHandler: Handle node behavior
    * Disable train/query model button when train/query node has not yet been placed
    * Enable train/query model button when train/query node has been placed
    * Disable input and selection of node boxes when node is not selected
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
                else if(targetNode.querySelector("#query-model") != null){
                    targetNode.querySelector("#query-model").disabled = false;
                }
                else{
                    document.querySelector("#query-model").disabled = true;
                    document.querySelector("#train-model").disabled = true;
                }  

                if(targetNode.querySelector("#input-number")){
                    targetNode.querySelector("#x").disabled = false;
                    targetNode.querySelector("#y").disabled = false;
                }
                if(targetNode.querySelector("#input-number-file")){
                    targetNode.querySelector("#x-file").disabled = false;
                    targetNode.querySelector("#y-file").disabled = false;
                }
                if(targetNode.querySelector("#input-number-logistic")){
                    targetNode.querySelector("#class").disabled = false;
                    targetNode.querySelector("#x-log").disabled = false;
                    targetNode.querySelector("#y-log").disabled = false;
                }
                if(targetNode.querySelector("#input-image")){
                    targetNode.querySelector("#img").disabled = false;
                }
                if(targetNode.querySelector("#input-text")){
                    targetNode.querySelector("#text").disabled = false;
                }
                if(targetNode.querySelector("#hidden-dense1")){
                    targetNode.querySelector("#units1").disabled = false;
                    targetNode.querySelector("#activationFunc1").disabled = false;
                }
                if(targetNode.querySelector("#hidden-dense2")){
                    targetNode.querySelector("#units2").disabled = false;
                    targetNode.querySelector("#activationFunc2").disabled = false;
                }
                if(targetNode.querySelector("#hidden-dense3")){
                    targetNode.querySelector("#units3").disabled = false;
                    targetNode.querySelector("#activationFunc3").disabled = false;
                }
                if(targetNode.querySelector("#recordButton")){
                    targetNode.querySelector("#recordButton").disabled = false;
                }
            }
        }
    };
    const observer = new MutationObserver(callback);
    observer.observe(targetNode, config);
});