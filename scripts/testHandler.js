/*
    testHandler: Handle test container
*/


document.addEventListener("DOMContentLoaded", function() {
    const targetNode = document.getElementById("constructModel");
    const config = {childList : true};

    const callback = (mutationList, observer) => {
        for(const mutation of mutationList){
            if(mutation.type === "childList"){
                if(targetNode.querySelector("#input-number") != null){
                    document.getElementById("num-test-page").style.display = "block";
                    document.getElementById("image-test-page").style.display = "none";
                    document.getElementById("text-test-page").style.display = "none";
                }
                else if(targetNode.querySelector("#input-image") != null){
                    document.getElementById("num-test-page").style.display = "none";
                    document.getElementById("image-test-page").style.display = "block";
                    document.getElementById("text-test-page").style.display = "none";
                }
                else if(targetNode.querySelector("#input-text") != null){
                    document.getElementById("num-test-page").style.display = "none";
                    document.getElementById("image-test-page").style.display = "none";
                    document.getElementById("text-test-page").style.display = "block";
                }
            }
        }
    };
    const observer = new MutationObserver(callback);
    observer.observe(targetNode, config);
});