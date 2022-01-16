var FILE_SELECTED = false;


$( "#submit" ).click(function() {
    var checked = $("#inputForm input:checked").length > 0;

    if(!FILE_SELECTED && !checked) {
        alert("Please select image file and check at least one option");
        return false;
    }
    else if(!FILE_SELECTED){
        alert("Please select image file");
        return false;
    }
    else if(!checked){
        alert("Please check at least one option has been selected");
        return false;
    }
});


$(document).on('change', '.custom-file-input', function () {
    let fileName = $(this).val().replace(/\\/g, '/').replace(/.*\//, '');
    $( "#fileInputLabel" ).text(fileName);
    FILE_SELECTED = true
    console.log(FILE_SELECTED)
});
