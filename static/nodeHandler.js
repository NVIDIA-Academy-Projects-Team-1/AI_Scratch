/*
    nodeHandler.js: Handle node behavior
    * Disable train/query model button when train/query node has not yet been placed
    * Enable train/query model button when train/query node has been placed
    * Disable input and selection node boxes when node is not selected
*/


document.addEventListener("DOMContentLoaded", function() {
    const targetNode = document.getElementById("constructModel");
    const config = {childList : true};

    const callback = (mutationList, observer) => {
        for(const mutation of mutationList){
            if(mutation.type === "childList"){
                // Train, query button
                if(targetNode.querySelector("#train-model")){
                    targetNode.querySelector("#train-model").disabled = false;
                }
                else if(targetNode.querySelector("#query-model")){
                    targetNode.querySelector("#query-model").disabled = false;
                }
                else{
                    document.querySelector("#query-model").disabled = true;
                    document.querySelector("#train-model").disabled = true;
                }  
                
                // Input number - linear regression
                if(targetNode.querySelector("#input-number")){
                    targetNode.querySelector("#x").disabled = false;
                    targetNode.querySelector("#y").disabled = false;
                }
                else{
                    document.querySelector("#x").disabled = true;
                    document.querySelector("#y").disabled = true;
                }

                // Input number txt file
                if(targetNode.querySelector("#input-number-file")){
                    targetNode.querySelector("#x-file").disabled = false;
                    targetNode.querySelector("#y-file").disabled = false;
                }
                else{
                    document.querySelector("#x-file").disabled = true;
                    document.querySelector("#y-file").disabled = true;
                }

                // Input CSV file
                if(targetNode.querySelector("#input-csv-file")){
                    targetNode.querySelector("#csv-file").disabled = false;
                }
                else{
                    document.querySelector("#csv-file").disabled = true;
                }

                // Input number - logistic regression
                if(targetNode.querySelector("#input-number-logistic")){
                    targetNode.querySelector("#class").disabled = false;
                    targetNode.querySelector("#x-log").disabled = false;
                    targetNode.querySelector("#y-log").disabled = false;
                }
                else{
                    document.querySelector("#class").disabled = true;
                    document.querySelector("#x-log").disabled = true;
                    document.querySelector("#y-log").disabled = true;
                }
                
                // Input image
                if(targetNode.querySelector("#input-image")){
                    targetNode.querySelector("#img").disabled = false;
                }
                else{
                    document.querySelector("#img").disabled = true;
                }

                // Input text
                if(targetNode.querySelector("#input-text")){
                    targetNode.querySelector("#text").disabled = false;
                }
                else{
                    document.querySelector("#text").disabled = true;
                }

                // Input audio
                if(targetNode.querySelector("#recordButton")){
                    targetNode.querySelector("#recordButton").disabled = false;
                }
                else{
                    document.querySelector("#recordButton").disabled = true;
                }

                // Dense layer 1
                if(targetNode.querySelector("#hidden-dense1")){
                    targetNode.querySelector("#units1").disabled = false;
                    targetNode.querySelector("#activationFunc1").disabled = false;
                }
                else{
                    document.querySelector("#units1").disabled = true;
                    document.querySelector("#activationFunc1").disabled = true;
                }

                // Dense layer 2
                if(targetNode.querySelector("#hidden-dense2")){
                    targetNode.querySelector("#units2").disabled = false;
                    targetNode.querySelector("#activationFunc2").disabled = false;
                }
                else{
                    document.querySelector("#units2").disabled = true;
                    document.querySelector("#activationFunc2").disabled = true;
                }

                // Dense layer 3
                if(targetNode.querySelector("#hidden-dense3")){
                    targetNode.querySelector("#units3").disabled = false;
                    targetNode.querySelector("#activationFunc3").disabled = false;
                }
                else{
                    document.querySelector("#units3").disabled = true;
                    document.querySelector("#activationFunc3").disabled = true;
                }
            }
        }
    };
    const observer = new MutationObserver(callback);
    observer.observe(targetNode, config);
});