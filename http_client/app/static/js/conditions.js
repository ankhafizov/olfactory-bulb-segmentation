$( "#submit" ).click(function() {
    var checked = $("#inputForm input:checked").length > 0;
    if (!checked){
        alert("Please check at least one option has been selected");
        return false;
    }
});


$(document).on('change', '.custom-file-input', function () {
    let fileName = $(this).val().replace(/\\/g, '/').replace(/.*\//, '');
    console.log($( "#fileInputLabel" ).value)
    $( "#fileInputLabel" ).text(fileName);
});


$("#submit").click(function() { 
    // bCheck is a input type button
    var fileName = $("#file1").val();

    if(fileName) { // returns true if the string is not empty
        alert(fileName + " was selected");
    } else { // no file was selected
        alert("no file selected");
    }
});