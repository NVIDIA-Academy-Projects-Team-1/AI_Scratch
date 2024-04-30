function getnumbers(){
    const x = document.getElementById('x').value;
    const y = document.getElementById('y').value;
}
function getimages(att_zone,btn){
    /* att_zone : 이미지들이 들어갈 위치 id, btn : file tag id */
    var attZone = document.getElementById(att_zone);
    var btnAtt = document.getElementById(btn);
    var sel_files = [];

    // 이미지와 체크 박스를 감싸고 있는 div 속성
    var div_style = 'display:inline-block;position:relative;'
                  + 'width:10px;height:10px;margin:10px:border:1px solid #00f;z-index:1';
    // 미리보기 이미지 속성
    var img_style = 'width:100%;height:100%;z-index:none';
    // 이미지안에 표시되는 체크박스의 속성
    var chk_style = 'width:10px;height:10px;position:absolute;font-size:24px;'
                  + 'rigth:0px;bottom:0px;z-index:999;background-color:rgba(255,255,255,0.1);color:#f00';

    btnAtt.onchange = function(img){
        var files = img.target.files;
        var fileArr = Array.prototype.slice.call(files);
        for(f of fileArr){
            imageLoader(f);        
        }
    }
    //탐색기에서 끌어서놓기 사용
    attZone.addEventListener('dragenter', function(img){
        img.preventDefault();
        img.stopPropagation();
    },false)
    attZone.addEventListener('dragover', function(img){
        img.preventDefault();
        img.stopPropagation();
    },false)
    attZone.addEventListener('drop', function(img){
        var files = {};
        img.preventDefault();
        img.stopPropagation();
        var dt = img.dataTransfer;
        files = dt.files;
        for(f of files){
            imageLoader(f);
        }
    },false)

    // 첨부된 이미지를 넣고 미리보기
    imageLoader = function(file){
        sel_files.push(file);
        var reader = new FileReader();
        reader.onload = function(img_pack){
            let image = document/createElement('image');
            image.setAttribute('style',img_style);
            image.src = img_pack.target.result;
            attZone.appendChild(makeDiv(image, file));
        }
        reader.readAsDataURL(file);
    }
    // 첨부된 파일이 있는경우 checkbox & attzone 에 추가할 div 만들어 반환
    makeDiv = function(image,file){
        var div = document.createElement('div');
        div.setAttribute('style',div_style);
        var btn = document.createElement('input');
        btn.setAttribute('type','button');
        btn.setAttribute('value','x');
        btn.setAttribute('delFile',file_name);
        btn.setAttribute('style',chk_style);
        btn.onclick = function(ev){
            var ele = ev.srcElement;
            var delFile = ele.getAttribute('delFile');
            for(var i = 0; i<sel_files.length; i++){
                if(delFile == sel_files[i].name){
                    sel_files.splice(i,1);
                }
            }
            dt = new DataTransfer();
            for(f in sel_files){
                var file = sel_files[f];
                dt.items.add(file);
            }
            btnAtt.files = dt.files;
            var p = ele.parentNode;
            attZone.removeChild(p)
        }
    }
}('att_zone','btnAtt')