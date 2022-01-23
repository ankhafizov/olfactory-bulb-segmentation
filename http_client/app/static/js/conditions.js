var FILE_SELECTED = false;


$("#submit").click(function () {
    var anything_checked = $("#inputForm input:checked").length > 0;

    if (!FILE_SELECTED && !anything_checked) {
        alert("Please select image file and check at least one option");
        return false;
    }
    else if (!FILE_SELECTED) {
        alert("Please select image file");
        return false;
    }
    else if (!anything_checked) {
        alert("Please check at least one option has been selected");
        return false;
    }

    $("#progressLoading").removeClass('d-none');
});


$(document).on('change', '.custom-file-input', function () {
    let fileName = $(this).val().replace(/\\/g, '/').replace(/.*\//, '');
    $("#fileInputLabel").text(fileName);
    FILE_SELECTED = true
    console.log(FILE_SELECTED)
});


// keep checks after refresh
$(function () {
    var data = localStorage.showning;
    $("input[name='switch_show']")
      .prop('checked',data=='true')
      .change(function () {
         localStorage.showning = $(this).prop("checked");
      });
});
