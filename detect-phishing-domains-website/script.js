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
        parentTitle.textContent = `Parent Domain: ${parent}`;
        parentElement.appendChild(parentTitle);

        const childrenTable = document.createElement('table');
        childrenTable.classList.add('children-table');
        childrenTable.classList.add('bordered'); // Add bordered class for styling
        
        // Create table header
        const headerRow = document.createElement('tr');
        const headers = ['Child Domain', 'Domain Similarity (%)', 'Content Similarity (%)',  'Title Similarity (%)'];
        headers.forEach(headerText => {
            const headerCell = document.createElement('th');
            headerCell.textContent = headerText;
            headerRow.appendChild(headerCell);
        });
        childrenTable.appendChild(headerRow);

        // Add data rows
        data[parent].forEach(child => {
            const childRow = document.createElement('tr');
            const childData = [
                shortenURL(child[0]), // Shortened Child Domain
                child[1].domain_similarity === -1 ? 'NA' : child[1].domain_similarity.toFixed(2), // Domain Similarity
                child[1].content_similarity === -1 ? 'NA' : child[1].content_similarity.toFixed(2), // Content Similarity
                child[1].title_similarity === -1 ? 'NA' : child[1].title_similarity.toFixed(2) // Title Similarity
            ];

            // Add background color based on thresholds
            childRow.style.backgroundColor = getRowColor(child[1]);

            childData.forEach(cellData => {
                const cell = document.createElement('td');
                cell.textContent = cellData;
                childRow.appendChild(cell);
            });
            childrenTable.appendChild(childRow);
        });        

        parentElement.appendChild(childrenTable);
        resultsContainer.appendChild(parentElement);
    }
}

function getRowColor(similarityData) {
    // Define your threshold values here
    const domainThreshold = 90;
    const contentThreshold = 80;
    const titleThreshold = 80;

    // Determine which threshold to use based on available data
    const domainSimilarity = similarityData.domain_similarity !== -1 ? similarityData.domain_similarity : 0;
    const contentSimilarity = similarityData.content_similarity !== -1 ? similarityData.content_similarity : 0;
    const titleSimilarity = similarityData.title_similarity !== -1 ? similarityData.title_similarity : 0;

    // Determine the color based on the threshold
    if (domainSimilarity >= domainThreshold || contentSimilarity >= contentThreshold || titleSimilarity >= titleThreshold) {
        return 'red'; // or any color you prefer for a high similarity
    } else if (domainSimilarity >= 50 || contentSimilarity >= 50 || titleSimilarity >= 50) {
        return 'yellow'; // or any color you prefer for moderate similarity
    } else {
        return 'green'; // or any color you prefer for low similarity
    }
}


function shortenURL(url) {
    const maxLength = 40; // Maximum length for the shortened URL
    if (url.length <= maxLength) {
        return url;
    }
    return url.substr(0, maxLength - 3) + '...'; // Truncate URL and add ellipsis
}

