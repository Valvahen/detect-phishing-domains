from flask import jsonify, request, Flask
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
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import textdistance
import nltk
import subprocess
import os
from urllib.parse import urlparse
import re

# Download NLTK tokenizer data
nltk.download('punkt')

# Disable SSL certificate verification
session = requests.Session()
session.verify = False

img_out_dir = "./favicons_bl/"

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

def calculate_similarity(paragraph1, paragraph2, n=2):
    if not paragraph1 or not paragraph2:
        return -1
    try:
        # Tokenize the paragraphs
        tokens_paragraph1 = word_tokenize(paragraph1.lower())
        tokens_paragraph2 = word_tokenize(paragraph2.lower())
        
        # Join tokens into strings of n-grams
        ngrams_paragraph1 = [" ".join(tokens_paragraph1[i:i+n]) for i in range(len(tokens_paragraph1)-n+1)]
        ngrams_paragraph2 = [" ".join(tokens_paragraph2[i:i+n]) for i in range(len(tokens_paragraph2)-n+1)]
        
        # Concatenate paragraphs
        combined_paragraphs = [paragraph1, paragraph2]
        
        # Compute TF-IDF scores
        tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenize, lowercase=True, norm=None)
        tfidf_matrix = tfidf_vectorizer.fit_transform(combined_paragraphs)
        
        # Extract TF-IDF vectors for each paragraph
        tfidf_paragraph1 = tfidf_matrix[0]
        tfidf_paragraph2 = tfidf_matrix[1]
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_paragraph1, tfidf_paragraph2)[0][0]
        
        # Normalize similarity score to percentage
        similarity_percentage = similarity * 100
        
        return similarity_percentage
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return -1

# Function to fetch and cache favicon
def get_favicon(url):
    url = "https://" + url
    if url in favicon_cache:
        return favicon_cache[url]
    try:
        icons = favicon.get(url)
        if not icons:
            favicon_cache[url] = None
            return None
        favicon_url = icons[0].url
        response = requests.get(favicon_url)
        image = Image.open(BytesIO(response.content))
        
        # Resize favicon to a standard size for comparison
        image = image.resize((32, 32))
        
        # Convert to grayscale for structural similarity comparison
        image = image.convert('L')
        
        # Save favicon temporarily for comparison
        filename = f"favicons/favicon_{url.replace('http://', '').replace('https://', '').replace('/', '_')}.png"
        image.save(filename)
        
        favicon_cache[url] = filename
        return filename
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
        
        # Resize image to a standard size for comparison
        image = image.resize((32, 32))
        
        # Convert to grayscale for structural similarity comparison
        image = image.convert('L')
        
        # Save image temporarily for comparison
        filename = f"images/image_{url.replace('http://', '').replace('https://', '').replace('/', '_')}.png"
        image.save(filename)
        
        favicon_cache[url] = filename
        return filename
    except Exception as e:
        print(f"Error fetching image from {url}: {e}")
        return None

import subprocess

def compare_images(parent_image, child_image):
    try:
        # Define the command to compare images using ImageMagick
        command = f"magick compare -metric RMSE {parent_image} {child_image} null: 2>&1"
        
        # Execute the command and capture the output
        output = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        # Check if stdout exists
        if hasattr(output, 'stdout') and output.stdout:
            # Extract similarity metric from within parentheses
            similarity_match = re.search(r'\((.*?)\)', output.stdout)
            if similarity_match:
                similarity_str = similarity_match.group(1)
                similarity_rmse = float(similarity_str)
            else:
                raise ValueError("Similarity metric not found in output")
            
            normalized_rmse = similarity_rmse * 100
            
            # Calculate similarity percentage (higher values indicate more similarity)
            similarity_percentage = 100 - normalized_rmse
            
            return similarity_percentage
        else:
            # If stdout doesn't exist or is empty, print the stderr output for debugging purposes
            print("Error occurred:", output.stderr)
            return -1
    except Exception as e:
        print("Error comparing images:", e)
        return -1
    
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

def compare_titles(title1, title2, n=2):
    if not title1 or not title2:
        return -1
    try:
        # Calculate Damerau-Levenshtein distance between titles
        similarity_score = textdistance.damerau_levenshtein.normalized_similarity(title1.lower(), title2.lower())
        
        # Normalize the similarity score to a percentage
        similarity_percentage = similarity_score * 100
        
        return similarity_percentage
    except Exception as e:
        print(f"Error comparing titles: {e}")
        return -1

def calculate_domain_similarity(parent, child):
    if not parent or not child:
        return -1
    try:
        # Convert domain strings to lowercase
        parent_lower = parent.lower()
        child_lower = child.lower()

        # Calculate positional Jaccard similarity
        parent_set = set(parent_lower)
        child_set = set(child_lower)
        intersection_count = len(parent_set.intersection(child_set))
        union_count = len(parent_set.union(child_set))

        # Calculate similarity percentage
        similarity_percentage = (intersection_count / union_count) * 100

        return similarity_percentage
    except Exception as e:
        print(f"Error calculating domain similarity: {e}")
        return -1
    
def download_favicon(url, filename=None):
    ensure_http(url)
    parsed_url = urlparse(url)

    if not filename:
        # use second-level domain (SLD) for filename
        filename = parsed_url.netloc
    # check if favicon already exists
    favicon_output_filename = img_out_dir + filename + ".ico"
    if os.path.exists(favicon_output_filename):
        print(favicon_output_filename + " already exists!")
        return

    # get url without path
    url = parsed_url.scheme + "://" + parsed_url.netloc
    print(url)
    response = session.get(url)

    # parse and get the favicon URL from the HTML content
    soup = BeautifulSoup(response.content, "html.parser")
    favicon_url = get_favicon_url_from_html(soup, url)

    if favicon_url:
        # download the favicon
        response = session.get(favicon_url)
        with open(favicon_output_filename, "wb") as f:
            f.write(response.content)
    else:
        print("Could not find favicon URL")


def get_favicon_url_from_html(soup, url):
    favicon_url = None
    for link in soup.find_all("link", {"rel": ["shortcut icon", "icon"]}):
        favicon_url = link.get("href")
        break
    if favicon_url and not favicon_url.startswith("http"):
        favicon_url = url + favicon_url

    return favicon_url


def download_favicons(links):
    for link in links:
        formatted_link = ensure_http(link)
        download_favicon(formatted_link)

@app.route('/', methods=['POST'])
def detect_phishing():
    file = request.files['file']
    child_domains = file.read().decode('utf-8').splitlines()
    
    parent_data = pd.read_csv("whitelist2.csv")
    parent_domains = parent_data['domain'].values

    threshold_ratio = 0
    parent_child_dict = {}

    download_favicons(child_domains)

    for parent in parent_domains:
        matching_children = []
        for child in child_domains:
            ratio = fuzz.ratio(parent, child)
            if ratio >= threshold_ratio:
                try:
                    # Calculate domain name similarity
                    domain_similarity = calculate_domain_similarity(parent, child)

                    # Fetch content from parent and child domains
                    parent_content = extract_website_content(parent)
                    child_content = extract_website_content(child)

                    # Calculate text similarity
                    content_similarity = calculate_similarity(parent_content, child_content) if parent_content and child_content else 0.0

                    # Fetch and compare favicons
                    # parent_favicon_url = get_favicon(parent)
                    # child_favicon_url = get_favicon(child)
                    parent_favicon_location = "./favicons_wl/" + parent + ".ico"
                    child_favicon_location = "./favicons_bl/" + child + ".ico"
                    favicon_similarity = 0.0
                    if os.path.isfile(parent_favicon_location) and os.path.isfile(child_favicon_location):
                        try:
                            favicon_similarity = compare_images(parent_favicon_location, child_favicon_location)
                            print("Favicon Similarity:", favicon_similarity)
                        except Exception as e:
                            print(f"Error comparing favicons for {parent} and {child}: {e}")
                            favicon_similarity = -1  # Set favicon similarity to 'NA' when an error occurs
                    else:
                        print("One or both favicon files do not exist.")
                        favicon_similarity = -1
                    # Fetch and compare titles
                    parent_title = get_title(parent)
                    child_title = get_title(child)
                    title_similarity = compare_titles(parent_title, child_title) if parent_title and child_title else 0.0

                    # Append all individual similarities
                    matching_children.append((child, {
                        'domain_similarity': domain_similarity,
                        'content_similarity': content_similarity,
                        'favicon_similarity': favicon_similarity,
                        'title_similarity': title_similarity,
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
