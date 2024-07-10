document.getElementById('upload-form').addEventListener('submit', function (e) {
    e.preventDefault();

    const fileInput = document.getElementById('file-input');
    if (fileInput.files.length === 0) {
        alert('Please select a file to upload.');
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    // Collect selected features
    const featureCheckboxes = document.querySelectorAll('input[name="features"]:checked');
    const selectedFeatures = [];
    featureCheckboxes.forEach(checkbox => {
        selectedFeatures.push(checkbox.value);
    });
    formData.append('features', JSON.stringify(selectedFeatures));

    // Show loading bar
    const loading = document.getElementById('loading');
    loading.style.display = 'block';

    // Hide previous results and errors
    const resultsContainer = document.getElementById('results');
    const filteredresultsContainer = document.getElementById('filtered-results');
    const errorMessage = document.getElementById('error-message');
    const timeTaken = document.getElementById('time-taken');
    resultsContainer.innerHTML = '';
    filteredresultsContainer.innerHTML = '';
    errorMessage.textContent = '';
    timeTaken.textContent = '';

    const startTime = Date.now(); // Start time

    fetch('http://127.0.0.1:5000/', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Hide loading indicator
        loading.style.display = 'none';

        // Check if data contains a link to download CSV
        if (data.csv_download_link) {
            const downloadLink = document.createElement('a');
            downloadLink.href = data.csv_download_link;
            downloadLink.textContent = 'Download CSV';
            downloadLink.download = data.filename;

            const processeddownloadLink = document.createElement('a');
            processeddownloadLink.href = data.processed_file_download_link;
            processeddownloadLink.textContent = 'Download Processed CSV';
            processeddownloadLink.download = data.filename;

            resultsContainer.appendChild(downloadLink);
            filteredresultsContainer.appendChild(processeddownloadLink)
        } else {
            console.error('Error: No CSV download link found in response.');
            errorMessage.textContent = 'Error: No CSV download link found.';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        errorMessage.textContent = `Error: ${error.message}`;
        loading.style.display = 'none'; // Hide loading bar
    });
});
