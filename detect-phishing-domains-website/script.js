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
    const errorMessage = document.getElementById('error-message');
    const timeTaken = document.getElementById('time-taken');
    resultsContainer.innerHTML = '';
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
        displayResults(data);
        const endTime = Date.now(); // End time
        const timeElapsed = (endTime - startTime) / 1000; // Time in seconds
        timeTaken.textContent = `Time taken: ${timeElapsed.toFixed(2)} seconds`;
  
        loading.style.display = 'none'; // Hide loading bar
    })
    .catch(error => {
        console.error('Error:', error);
        errorMessage.textContent = `Error: ${error.message}`;
        loading.style.display = 'none'; // Hide loading bar
    });
});
  
function displayResults(data) {
    const resultsContainer = document.getElementById('results');
    resultsContainer.innerHTML = '';

    for (const parent in data) {
        const parentElement = document.createElement('div');
        parentElement.classList.add('result-item');

        const parentTitle = document.createElement('h3');
        parentTitle.textContent = `Target Domain: ${parent}`;
        parentElement.appendChild(parentTitle);

        const childrenTable = document.createElement('table');
        childrenTable.classList.add('children-table');
        childrenTable.classList.add('bordered');
        
        // Create table header
        const headerRow = document.createElement('tr');
        const headers = ['Phishing Domain'];
        const firstChild = data[parent][0];
        if (firstChild[1].domain_similarity !== undefined) headers.push('Domain Similarity (%)');
        if (firstChild[1].content_similarity !== undefined) headers.push('Content Similarity (%)');
        if (firstChild[1].title_similarity !== undefined) headers.push('Title Similarity (%)');
        if (firstChild[1].screenshot_similarity !== undefined) headers.push('Screenshot Similarity (%)');
        headers.forEach(headerText => {
            const headerCell = document.createElement('th');
            headerCell.textContent = headerText;
            headerRow.appendChild(headerCell);
        });
        childrenTable.appendChild(headerRow);

        // Add data rows
        data[parent].forEach(child => {
            const childData = [shortenURL(child[0])];
            if (child[1].domain_similarity !== undefined) childData.push(child[1].domain_similarity === -1 ? 'NA' : child[1].domain_similarity.toFixed(2));
            if (child[1].content_similarity !== undefined) childData.push(child[1].content_similarity === -1 ? 'NA' : child[1].content_similarity.toFixed(2));
            if (child[1].title_similarity !== undefined) childData.push(child[1].title_similarity === -1 ? 'NA' : child[1].title_similarity.toFixed(2));
            if (child[1].screenshot_similarity !== undefined) childData.push(child[1].screenshot_similarity === -1 ? 'NA' : child[1].screenshot_similarity.toFixed(2));
            
            // Check if any similarity meets the threshold of 60%
            const showRow = childData.slice(1).some(similarity => {
                const similarityValue = parseFloat(similarity);
                return similarityValue >= 0;
            });

            if (showRow) {
                const childRow = document.createElement('tr');
                childData.forEach((cellData, index) => {
                    const cell = document.createElement('td');
                    cell.textContent = cellData;
                    if (index !== 0) { // Exclude the first cell (Phishing Domain)
                        const similarityValue = parseFloat(cellData);
                        if (similarityValue >= 70 && similarityValue < 85) {
                            cell.classList.add('pale-yellow');
                        } else if (similarityValue >= 85) {
                            cell.classList.add('pale-red');
                        }
                    }
                    childRow.appendChild(cell);
                });
                childrenTable.appendChild(childRow);
            }
        });

        if (childrenTable.rows.length > 1) { // Check if there are rows (excluding header)
            parentElement.appendChild(childrenTable);
            resultsContainer.appendChild(parentElement);
        }
    }
}

function shortenURL(url) {
    const maxLength = 40; // Maximum length for the shortened URL
    if (url.length <= maxLength) {
        return url;
    }
    return url.substr(0, maxLength - 3) + '...'; // Truncate URL and add ellipsis
}
