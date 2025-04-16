document.getElementById('itemForm').addEventListener('submit', function (e) {
    e.preventDefault();

    var itemName = document.getElementById('itemName').value;
    var itemPrice = document.getElementById('itemPrice').value;

    // Show loading or feedback before actual prediction
    alert('Getting seasonal prediction for ' + itemName + '...');

    $.ajax({
        url: '/item_predict',
        type: 'POST',
        data: {
            itemName: itemName,
            itemPrice: itemPrice
        },
        success: function (response) {
            // Display the prediction summary and recommendation
            document.getElementById('predictionSummary').innerHTML = `
                <p><strong>Prediction:</strong> ${response.summary}</p>
                <p><strong>Recommendation:</strong> ${response.recommendation}</p>
            `;
        },
        error: function (error) {
            alert("Error in getting prediction");
        }
    });
});
document.getElementById('uploadForm').addEventListener('submit', function (e) {
    e.preventDefault();

    var fileInput = document.getElementById('fileInput');
    if (!fileInput.files.length) {
        alert('Please upload a file!');
        return;
    }

    // Show loading or feedback before actual prediction
    alert('Dataset uploaded! Processing predictions...');

    var formData = new FormData(this);  // Collect the form data

    $.ajax({
        url: '/predict',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {

            // Inject image paths
            document.getElementById('salesPlot').src = response.sales_plot;
            document.getElementById('decompositionPlot').src = response.decomposition_plot;
            document.getElementById('arimaForecastPlot').src = response.arima_forecast_plot;

            // Show the result card
            document.getElementById('resultCard').style.display = 'block';

            // Optional: Add summary info
            document.getElementById('predictionSummary').innerHTML = `
                <p><strong>Predicted Sales:</strong> ${response.summary}</p>
                <p><strong>Prediction Confidence:</strong> ${response.confidence}</p>
            `;
        },
        error: function (error) {
            alert("Error in file upload or prediction");
        }
    });
});
// Function to render the prediction chart after the data is received from the server
function renderPredictionChart(predictionData) {
    var ctx = document.getElementById('predictionChart').getContext('2d');
    var chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: predictionData.labels, // X-axis: Labels (e.g., Months, Time, etc.)
            datasets: [{
                label: 'Predicted Sales',
                data: predictionData.values, // Y-axis: Predicted Values
                borderColor: 'rgba(75, 192, 192, 1)',
                fill: false
            }]
        },
        options: {
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Sales'
                    }
                }
            }
        }
    });
}

// Example for how to handle prediction results
function displayPredictionResults(results) {
    // Show result card with prediction summary and confidence
    document.getElementById('resultCard').style.display = 'block';
    document.getElementById('predictionSummary').innerHTML = `
        <p><strong>Predicted Sales:</strong> ${results.summary}</p>
        <p><strong>Prediction Confidence:</strong> ${results.confidence}</p>
    `;


}
