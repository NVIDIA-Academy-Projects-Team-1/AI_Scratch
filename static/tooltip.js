var tooltipText = "<p class='text-left' style='font-size: 14px;'>1. 딥러닝에서의 뉴런은 마치 우리 뇌 안의 작은 친구들이야. 이 작은 친구들은 정보를 받고, 가공하고, 다시 보내는 역할을 해.<br>2. 시그모이드 함수는 입력된 숫자를 조금 이상한 숫자로 바꿔줘. 이 함수는 0과 1 사이의 숫자로 변환시켜주는데, 이렇게 하면 우리가 원하는 것을 쉽게 알 수 있어.<br>3. 소프트맥스 함수는 선택지 중에서 하나를 고르는 것을 도와주는 함수야. 예를 들어, 과일 중에서 제일 좋은 과일을 고를 때 사용돼!<br>4. ReLU 함수는 뉴런이들이 일할지 말지를 결정하는 스위치 역할을 해.</p>";
var tooltipImg = "<p class='text-left' style='font-size: 14px;'>1. 이미지 분류기는 우리가 원하는 사진을 올려서 그 사진에 무엇이 있는지를 컴퓨터가 찾아서 알려주는 도구입니다.<br>2. 예시로 강아지 사진을 이미지 분류기에 넣으면 그 강아지가 무슨 종류 인지 그리고 그 종류 예측을 얼마나의 정확히 했는지 결과가 나옵니다. (최대 1000개 종류까지 분류 할수 있습니다.) <br>3. 번역하는 과정에 잘못된 문장이 나올수도 있습니다. 허나 이는 추후에 더 좋은 컴퓨터와 기술이 개발되면 발전될수 있습니다.</p>";
var chatbot = "<p class='test-left' style='font-size:14px;'>1. 지금 세상에서 가장 많은 관심을 받고 있는 AI (인공지능) 기술입니다. 사용자가 문장이나 단어를 입력했을떄 그 입력돤 문장이나 단어에 기반하여 답을 생성해 제공하는 기술입니다.<br>2. 예시로 '안녕, 너는 누구니?' 라는 문장을 입력하면 '안녕하세요! 저는 AI입니다. 당신의 질문이나 도움이 필요한 사항을 알고 있습니다. 무엇을 도와드릴까요?' 같이 문장을 답하게 됩니다.</p>"

var tooltips = {
    "hidden-dense1": tooltipText,
    "hidden-dense2": tooltipText,
    "hidden-dense3": tooltipText,
    'hidden-image': tooltipImg,
    'hidden-chat': chatbot
};


document.addEventListener("DOMContentLoaded", function() {
    document.querySelectorAll('#hidden-dense1, #hidden-dense2, #hidden-dense3, #hidden-image, #hidden-chat').forEach(function(element) {
        element.addEventListener('mouseover', function() {
            document.getElementById('layer-description').innerHTML = tooltips[element.id];
        });

        element.addEventListener('mouseout', function() {
            document.getElementById('layer-description').innerHTML = ""; 
        });
    });
});