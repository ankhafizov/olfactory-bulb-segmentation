$( "#submit" ).click(function() {
    var checked = $("#frmText input:checked").length > 0;
    if (!checked){
        alert("Please check at least one option has been selected");
        return false;
    }
});