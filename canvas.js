document.querySelectorAll('.input-layer, .hidden-layer, .run-layer').forEach(node => {
    node.addEventListener('click', () => {
        
        const content = node.innerText;
        
        drawOnCanvas(content);
    });
});


function drawOnCanvas(content) {
    
    const canvas = document.getElementById('model-canvas');
    const context = canvas.getContext('2d');
    
    context.clearRect(0, 0, canvas.width, canvas.height);
    
    context.font = '20px Arial';
    context.fillText(content, 10, 30);
}