function denseFunction(){

    const spawn = requires('child_process').spawn;
    const units = requires(document.getElementById('units').value);
    const input_shape = document.getElementById('input_shape').value;
    const activationFunc = requires(document.getElementById('activationFunc').value);
    if(input_shape == undefined){
        const result = spawn('python',['denseFunction.py',units,activationFunc]);
    }
    else{
        const result = spawn('python',['denseFunction.py',units,input_shape,activationFunc]);
    }
    result.stdout.on('data',(result)=>{
        console.log(result.toString());
    });
}