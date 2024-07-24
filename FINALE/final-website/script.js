document.addEventListener('DOMContentLoaded', function () {
    const sourceRadios = document.querySelectorAll('input[name="source"]');
    const nixiFields = document.getElementById('nixi-fields');

    function toggleFields() {
        const selectedSource = document.querySelector('input[name="source"]:checked').value;
        if (selectedSource === 'nixi') {
            nixiFields.style.display = 'block';
        } else {
            nixiFields.style.display = 'none';
        }
    }

    // Initial check
    toggleFields();

    // Add event listeners to radio buttons
    sourceRadios.forEach(radio => {
        radio.addEventListener('change', toggleFields);
    });

    document.getElementById('upload-form').addEventListener('submit', function (e) {
        e.preventDefault();

        const selectedSource = document.querySelector('input[name="source"]:checked').value;
        const fileInput = document.getElementById('file-input');
        const dateInput = document.getElementById('date-input');

        const file = fileInput ? fileInput.files[0] : null;
        const formData = new FormData();
        if (file) {
            formData.append('file', file);
        }

        // Collect selected features
        const featureCheckboxes = document.querySelectorAll('input[name="features"]:checked');
        const selectedFeatures = [];
        featureCheckboxes.forEach(checkbox => {
            selectedFeatures.push(checkbox.value);
        });
        formData.append('features', JSON.stringify(selectedFeatures));

        // Collect date input value if 'nixi' is chosen
        if (selectedSource === 'nixi') {
            const date = dateInput ? dateInput.value : '';
            formData.append('date', JSON.stringify(date));
        } else {
            formData.append('date', JSON.stringify('')); // Provide an empty value for 'whois'
        }

        // Collect selected source value
        formData.append('source', selectedSource);

        // Show loading bar
        const loading = document.getElementById('loading');
        loading.style.display = 'block';

        // Hide previous results and errors
        const resultsContainer = document.getElementById('results');
        const errorMessage = document.getElementById('error-message');
        const timeTaken = document.getElementById('time-taken');
        resultsContainer.innerHTML = '';
        errorMessage.textContent = '';
        timeTaken.textContent = '';

        const startTime = Date.now(); // Start time

        fetch('http://127.0.0.1:5000/detect-phishing', {
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

                resultsContainer.appendChild(downloadLink);
            } else {
                console.error('Error: No CSV download link found in response.');
                errorMessage.textContent = 'Error: No CSV download link found.';
            }

            const endTime = Date.now();
            const timeElapsed = ((endTime - startTime) / 1000).toFixed(2);
            timeTaken.textContent = `Time taken: ${timeElapsed} seconds`;
        })
        .catch(error => {
            console.error('Error:', error);
            errorMessage.textContent = `Error: ${error.message}`;
            loading.style.display = 'none'; // Hide loading bar
        });
    });
});
