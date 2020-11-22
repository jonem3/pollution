function renderResults(jason) {
    var timestamps = [];
    var data = [[], [], [], [], []]
    var qualities = []
    for (var quality in jason) {
        let count = 0;
        for (var timestamp in jason[quality]) {
            if (timestamps.length < 5) {
                timestamps.push(timestamp);
                data[count].push(timestamp);
            }
            data[count].push(jason[quality][timestamp]);
            count++;
        }

        qualities.push(quality);
    }
    var headers = "<th>Date</th>"
    qualities.forEach((item) => {
        headers = headers.concat("<th>", item, "</th>");
    });
    var table = "";
    table = table.concat("<table id='results_table' class='display'><thead><tr>", headers, "</tr></thead></table>");
    $("#results").html(table);
    $('#results_table').DataTable({
        data: data,
        paging: false
    })
}

function sendQuery() {
    var locn = $("#location_drop");
    var qualities = [];
    $(".air_index").each(function () {
        var checkbox = $(this);
        if (checkbox.is(':checked')) {
            qualities.push(checkbox.val())
        }
    });
    var locationId = (locn.find(":selected").val());
    if (locationId.length > 0 && qualities.length > 0) {
        var requestJson = {
            location: locationId,
            quality: qualities
        };
        console.log(requestJson)
        $.ajax({
            url: locn.attr("data-ajax-url"),
            type: "POST",
            data: JSON.stringify(requestJson),
            contentType: "application/json; charset=utf-8",
            dataType: "json"
        }).done(function (jason) {
            console.log(jason);
            renderResults(jason);
        });
    } else {
        console.log("No location selected OR No qualities selected")
    }

}

$(document).ready(function () {
    $("#location_drop").change(sendQuery);
    $(".air_index").on('change', sendQuery);
});
