var tooltipText = "<p class='text'>딥러닝에서는 뉴런의 작동 원리를 모방하여 인공신경망을 구성합니다. 인공신경망은 여러 개의 뉴런(노드)과 뉴런 간의 연결(가중치)로 이루어져 있으며, 입력 신호가 네트워크를 통해 처리되고 출력으로 전달됩니다. 이를 통해 컴퓨터가 데이터를 학습하고 패턴을 인식할 수 있습니다</p>";

var tooltips = {
    "hidden-dense1": tooltipText,
    "hidden-dense2": tooltipText,
    "hidden-dense3": tooltipText
};

document.addEventListener("DOMContentLoaded", function() {
    document.querySelectorAll('#hidden-dense1, #hidden-dense2, #hidden-dense3').forEach(function(element) {
        element.addEventListener('mouseover', function() {
            document.getElementById('layer-description').innerHTML = tooltips[element.id];
        });

        element.addEventListener('mouseout', function() {
            document.getElementById('layer-description').innerHTML = ""; 
        });
    });
});