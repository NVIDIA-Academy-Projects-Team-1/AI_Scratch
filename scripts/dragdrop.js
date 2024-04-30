import Sortable from "https://cdn.skypack.dev/sortablejs@1.14.0";

var input_layers = document.getElementById('input-layers')
var architecture = document.querySelector("#arch")

// new Sortable.create(input_layers,{
//     group: {
//         name: "input-layers",
//         put: function(to) {
//             return to.el.children.length < 4;
//         }
//     },
//     animation: 100
// });

// new Sortable.create(architecture,{
//     group: {
//         name: "arch",
//         put: function(from) {
//             return from.el.children.length > 2 || 'clone';
//         }
//     },
//     animation: 100
// });

new Sortable.create(layer-1, {
    animation: 100
});


new Sortable.create(arch, {
    animation: 100
})










// app.js
// const item = document.querySelector('.item');
// const boxes = document.querySelectorAll('.box');

// item.addEventListener('dragstart', dragStart);

// boxes.forEach(box => {
//     box.addEventListener('dragenter', dragEnter)
//     box.addEventListener('dragover', dragOver);
//     box.addEventListener('dragleave', dragLeave);
//     box.addEventListener('drop', drop);
// });

// function dragStart(e) {
//     e.dataTransfer.setData('text/plain', e.target.id);

//     setTimeout(() => {
//         e.target.classList.add('hide');
//     }, 0);
// }

// function dragEnter(e) {
//     e.preventDefault();
//     e.target.classList.add('drag-over');
// }

// function dragOver(e) {
//     e.preventDefault();
//     e.target.classList.add('drag-over');
// }

// function dragLeave(e) {
//     e.target.classList.remove('drag-over');
// }

// function drop(e) {
//     e.target.classList.remove('drag-over');

//     const id = e.dataTransfer.getData('text/plain');
//     const draggable = document.getElementById(id);

//     e.target.appendChild(draggable);

//     draggable.classList.remove('hide');
// }