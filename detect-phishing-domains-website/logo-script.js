document.getElementById('logo-upload-form').addEventListener('submit', function (e) {
    e.preventDefault();

    const fileInput = document.getElementById('logo-file-input');
    if (fileInput.files.length === 0) {
        alert('Please select a file to upload.');
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    const loading = document.getElementById('logo-loading');
    loading.style.display = 'block';

    const errorMessage = document.getElementById('logo-error-message');
    errorMessage.textContent = '';

    fetch('http://127.0.0.1:5000/detect_logos', {
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

            const resultsContainer = document.getElementById('logo-results');
            resultsContainer.innerHTML = '';
            resultsContainer.appendChild(downloadLink);
        } else {
            console.error('Error: No CSV download link found in response.');
            errorMessage.textContent = 'Error: No CSV download link found.';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        errorMessage.textContent = `Error: ${error.message}`;
        loading.style.display = 'none';
    });
});
