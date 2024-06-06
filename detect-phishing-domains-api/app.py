from flask import jsonify, request, Flask
import flask
from flask_cors import CORS
import pandas as pd
from thefuzz import fuzz
import requests
import favicon
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from skimage.metrics import structural_similarity as ssim
import numpy as np

app = Flask(__name__)
CORS(app)

# Define dictionaries to store cached data
content_cache = {}
title_cache = {}
favicon_cache = {}

# Function to ensure URLs start with http:// or https://
def ensure_http(url):
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    return url

# Function to extract website content, with caching
def extract_website_content(url):
    url = ensure_http(url)
    if url in content_cache:
        return content_cache[url]
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            content = response.text
            content_cache[url] = content
            return content
        else:
            print(f"Failed to fetch content from {url}. Status code: {response.status_code}")
            return ''
    except requests.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return ''

def calculate_similarity(paragraph1, paragraph2):
    if not paragraph1 or not paragraph2:
        return 0.0
    sentences = [paragraph1, paragraph2]
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        cosine_sim = (tfidf_matrix * tfidf_matrix.T).A
        similarity_percentage = cosine_sim[0, 1] * 100
        return max(0.0, min(similarity_percentage, 100.0))  # Ensure similarity is between 0 and 100
    except ValueError as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

# Function to fetch and cache favicon
def get_favicon(url):
    url = ensure_http(url)
    if url in favicon_cache:
        return favicon_cache[url]
    try:
        icons = favicon.get(url)
        if not icons:
            favicon_cache[url] = None
            return None
        favicon_url = icons[0].url
        favicon_cache[url] = favicon_url
        return favicon_url
    except Exception as e:
        print(f"Error fetching favicon from {url}: {e}")
        return None

# Function to fetch and cache image
def fetch_image(url):
    if url in favicon_cache:
        return favicon_cache[url]
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        favicon_cache[url] = image
        return image
    except Exception as e:
        print(f"Error fetching image from {url}: {e}")
        return None

def compare_images(image1, image2):
    try:
        image1 = image1.convert('L')  # Convert to grayscale
        image2 = image2.convert('L')  # Convert to grayscale
        
        # Convert images to numpy arrays
        image1_np = np.array(image1)
        image2_np = np.array(image2)
        
        # Resize images to the same size if necessary
        if image1_np.shape != image2_np.shape:
            image2_np = np.resize(image2_np, image1_np.shape)
        
        # Calculate SSIM between the two images
        similarity_index, _ = ssim(image1_np, image2_np, full=True)
        
        similarity_percentage = similarity_index * 100
        return max(0.0, min(similarity_percentage, 100.0))  # Ensure similarity is between 0 and 100
    except Exception as e:
        print(f"Error comparing images: {e}")
        return 0.0

# Function to fetch and cache title
def get_title(url):
    url = ensure_http(url)
    if url in title_cache:
        return title_cache[url]
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string.strip() if soup.title else 'No title found'
        title_cache[url] = title
        return title
    except Exception as e:
        print(f"Error fetching title from {url}: {e}")
        return 'No title found'

def compare_titles(title1, title2):
    similarity = fuzz.ratio(title1, title2)
    return max(0.0, min(similarity, 100.0))  # Ensure similarity is between 0 and 100

@app.route('/flask-version')
def show_flask_version():
    return f"Flask Version: {flask.__version__}"

@app.route('/', methods=['POST'])
def detect_phishing():
    file = request.files['file']
    child_domains = file.read().decode('utf-8').splitlines()
    
    parent_data = pd.read_csv("whitelist.csv")
    parent_domains = parent_data['domain'].values

    threshold_ratio = 0
    parent_child_dict = {}

    for parent in parent_domains:
        matching_children = []
        for child in child_domains:
            ratio = fuzz.ratio(parent, child)
            if ratio >= threshold_ratio:
                try:
                    # Fetch content from parent and child domains
                    parent_content = extract_website_content(parent)
                    child_content = extract_website_content(child)

                    # Calculate text similarity
                    content_similarity = calculate_similarity(parent_content, child_content) if parent_content and child_content else 0.0

                    # Fetch and compare favicons
                    parent_favicon_url = get_favicon(parent)
                    child_favicon_url = get_favicon(child)
                    favicon_similarity = 0.0
                    if parent_favicon_url and child_favicon_url:
                        parent_image = fetch_image(parent_favicon_url)
                        child_image = fetch_image(child_favicon_url)
                        if parent_image and child_image:
                            favicon_similarity = compare_images(parent_image, child_image)

                    # Fetch and compare titles
                    parent_title = get_title(parent)
                    child_title = get_title(child)
                    title_similarity = compare_titles(parent_title, child_title) if parent_title and child_title else 0.0

                    # Calculate overall similarity based on weights
                    # Adjust the weights according to priority
                    content_weight = 0.2
                    favicon_weight = 0.4
                    title_weight = 0.4

                    overall_similarity = (content_weight * content_similarity + 
                                          favicon_weight * favicon_similarity + 
                                          title_weight * title_similarity) / (content_weight + favicon_weight + title_weight)

                    matching_children.append((child, {
                        'content_similarity': content_similarity,
                        'favicon_similarity': favicon_similarity,
                        'title_similarity': title_similarity,
                        'overall_similarity': overall_similarity
                    }))
                except Exception as e:
                    error_message = f"Error processing {parent} and {child}: {e}"
                    matching_children.append((child, {'error': error_message}))
                    print(error_message)
        
        if matching_children:
            parent_child_dict[parent] = matching_children

    return jsonify(parent_child_dict)

if __name__ == "__main__":
    app.run()