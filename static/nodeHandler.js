/*
    nodeHandler: Handle node behavior
    * Disable train/query model button when train/query node has not yet been placed
    * Enable train/query model button when train/query node has been placed
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
            }
        }
    };
    const observer = new MutationObserver(callback);
    observer.observe(targetNode, config);
});