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
  
        const childrenList = document.createElement('ul');
        data[parent].forEach(child => {
            const childItem = document.createElement('li');
            childItem.innerHTML = `
                <strong>Child Domain:</strong> ${child[0]}<br>
                <strong>Content Similarity:</strong> ${child[1].content_similarity.toFixed(2)}%<br>
                <strong>Favicon Similarity:</strong> ${child[1].favicon_similarity.toFixed(2)}%<br>
                <strong>Title Similarity:</strong> ${child[1].title_similarity.toFixed(2)}%<br>
                <strong>Overall Similarity:</strong> ${child[1].overall_similarity.toFixed(2)}%
            `;
            childrenList.appendChild(childItem);
        });
  
        parentElement.appendChild(childrenList);
        resultsContainer.appendChild(parentElement);
    }
  }
  