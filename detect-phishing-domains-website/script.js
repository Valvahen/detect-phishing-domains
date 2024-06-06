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
        const headers = ['Child Domain', 'Content Similarity (%)', 'Favicon Similarity (%)', 'Title Similarity (%)', 'Overall Similarity (%)'];
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
                child[1].content_similarity === 'NA' ? 'NA' : child[1].content_similarity.toFixed(2), // Content Similarity
                child[1].favicon_similarity === 'NA' ? 'NA' : child[1].favicon_similarity.toFixed(2), // Favicon Similarity
                child[1].title_similarity === 'NA' ? 'NA' : child[1].title_similarity.toFixed(2), // Title Similarity
                child[1].overall_similarity === 'NA' ? 'NA' : child[1].overall_similarity.toFixed(2) // Overall Similarity
            ];
            childData.forEach(cellData => {
                const cell = document.createElement('td');
                cell.textContent = cellData;
                childRow.appendChild(cell);
            });
            // Add color based on overall similarity
            const overallSimilarity = child[1].overall_similarity;
            if (overallSimilarity >= 0 && overallSimilarity <= 30) {
                childRow.classList.add('light-green');
            } else if (overallSimilarity > 30 && overallSimilarity <= 60) {
                childRow.classList.add('yellow');
            } else if (overallSimilarity > 60) {
                childRow.classList.add('red');
            }
            childrenTable.appendChild(childRow);
        });

        parentElement.appendChild(childrenTable);
        resultsContainer.appendChild(parentElement);
    }
}

function shortenURL(url) {
    const maxLength = 40; // Maximum length for the shortened URL
    if (url.length <= maxLength) {
        return url;
    }
    return url.substr(0, maxLength - 3) + '...'; // Truncate URL and add ellipsis
}