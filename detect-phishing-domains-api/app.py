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
import ssl
from requests.exceptions import InvalidURL
import requests.exceptions

# Suppress SSL warnings
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Download NLTK tokenizer data
nltk.download('punkt')

# Disable SSL certificate verification
session = requests.Session()
session.verify = False

# img_out_dir = r"detect-phishing-domains-main\detect-phishing-domains-api\favicons_bl"

app = Flask(__name__)
CORS(app)

# Define dictionaries to store cached data
content_cache = {}
title_cache = {}
# favicon_cache = {}

# Function to ensure URLs start with http:// or https://
def ensure_http(url):
    if not url.startswith(('http://', 'https://')):
        # Try with https:// first
        https_url = 'https://' + url
        try:
            requests.get(https_url)
            return https_url
        except requests.RequestException:
            # If https:// fails, try http://
            http_url = 'http://' + url
            try:
                requests.get(http_url)
                return http_url
            except requests.RequestException:
                # If both fail, return original url with http:// prefix
                return http_url
    return url

def extract_website_content(url):
    url = ensure_http(url)
    if url in content_cache:
        return content_cache[url]
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            content = response.text
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find the main content of the webpage
            main_content = soup.find('main')  # You can adjust this according to the structure of the webpage
            
            if main_content:
                content = main_content.get_text(separator='\n')
            else:
                # Check if body tag exists
                if soup.body:
                    content = soup.body.get_text(separator='\n')
                else:
                    # If neither <main> nor <body> tag exists, return an empty string
                    content = ''
            # Clean up the extracted text
            content = clean_text(content)
            content_cache[url] = content
            print(f"Content fetched for {url}")
            return content
        else:
            print(f"Failed to fetch content from {url}. Status code: {response.status_code}")
            return "No content found"
    except requests.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return "No content found"


def clean_text(text):
    # Remove extra spaces and newlines
    text = ' '.join(text.split())
    return text

from langdetect import detect

def calculate_similarity(paragraph1, paragraph2, n=2):
    if paragraph1 == "No content found" or paragraph2 == "No content found":
        return -1
    try:

        # Detect languages of the paragraphs
        lang_paragraph1 = detect(paragraph1)
        lang_paragraph2 = detect(paragraph2)
        
        # Check if languages are similar
        if lang_paragraph1 != lang_paragraph2:
            return -1

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
# def get_favicon(url):
#     url = "https://" + url
#     if url in favicon_cache:
#         return favicon_cache[url]
#     try:
#         icons = favicon.get(url)
#         if not icons:
#             favicon_cache[url] = None
#             return None
#         favicon_url = icons[0].url
#         response = requests.get(favicon_url)
#         image = Image.open(BytesIO(response.content))
        
#         # Resize favicon to a standard size for comparison
#         image = image.resize((32, 32))
        
#         # Convert to grayscale for structural similarity comparison
#         image = image.convert('L')
        
#         # Save favicon temporarily for comparison
#         filename = f"favicons/favicon_{url.replace('http://', '').replace('https://', '').replace('/', '_')}.png"
#         image.save(filename)
        
#         favicon_cache[url] = filename
#         return filename
#     except Exception as e:
#         print(f"Error fetching favicon from {url}: {e}")
#         return None


# # Function to fetch and cache image
# def fetch_image(url):
#     if url in favicon_cache:
#         return favicon_cache[url]
#     try:
#         response = requests.get(url, stream=True)
#         response.raise_for_status()
#         image = Image.open(BytesIO(response.content))
        
#         # Resize image to a standard size for comparison
#         image = image.resize((32, 32))
        
#         # Convert to grayscale for structural similarity comparison
#         image = image.convert('L')
        
#         # Save image temporarily for comparison
#         filename = f"images/image_{url.replace('http://', '').replace('https://', '').replace('/', '_')}.png"
#         image.save(filename)
        
#         favicon_cache[url] = filename
#         return filename
#     except Exception as e:
#         print(f"Error fetching image from {url}: {e}")
#         return None

# import subprocess

# def compare_images(parent_image, child_image):
#     try:
#         # Define the command to compare images using ImageMagick
#         command = f"magick compare -metric RMSE {parent_image} {child_image} null: 2>&1"
        
#         # Execute the command and capture the output
#         output = subprocess.run(command, shell=True, capture_output=True, text=True)
        
#         # Check if stdout exists
#         if hasattr(output, 'stdout') and output.stdout:
#             # Extract similarity metric from within parentheses
#             similarity_match = re.search(r'\((.*?)\)', output.stdout)
#             if similarity_match:
#                 similarity_str = similarity_match.group(1)
#                 similarity_rmse = float(similarity_str)
#             else:
#                 raise ValueError("Similarity metric not found in output")
            
#             normalized_rmse = similarity_rmse * 100
            
#             # Calculate similarity percentage (higher values indicate more similarity)
#             similarity_percentage = 100 - normalized_rmse
            
#             return similarity_percentage
#         else:
#             # If stdout doesn't exist or is empty, print the stderr output for debugging purposes
#             print("Error occurred:", output.stderr)
#             return -1
#     except Exception as e:
#         print("Error comparing images:", e)
#         return -1
    
# Function to fetch and cache title
def get_title(url):
    url = ensure_http(url)
    if url in title_cache:
        return title_cache[url]
    try:
        # Attempt to fetch title with HTTPS
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string.strip() if soup.title else 'No title found'
        title_cache[url] = title
        print(f"title fetched for {url}")
        return title
    except Exception as e:
        print(f"Error fetching title from {url}: {e}")

        # If HTTPS fails, attempt with HTTP
        if url.startswith('https://'):
            http_url = 'http://' + url[len('https://'):]
            return get_title(http_url)
        return 'No title found'

def compare_titles(title1, title2, n=2):
    if title1 == 'No title found' or title2 == 'No title found':
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
    
import requests.exceptions

# def download_favicon(url, filename=None):
#     parsed_url = urlparse(url)

#     if not filename:
#         # use second-level domain (SLD) for filename
#         filename = parsed_url.netloc
#     # check if favicon already exists
#     favicon_output_filename = os.path.join(img_out_dir, filename + ".ico")
#     if os.path.exists(favicon_output_filename):
#         print(favicon_output_filename + " already exists!")
#         return

#     # get url without path
#     url = parsed_url.scheme + "://" + parsed_url.netloc
#     print(url)
#     try:
#         response = session.get(url)

#         # parse and get the favicon URL from the HTML content
#         soup = BeautifulSoup(response.content, "html.parser")
#         favicon_url = get_favicon_url_from_html(soup, url)

#         if favicon_url:
#             # download the favicon
#             response = session.get(favicon_url)
#             with open(favicon_output_filename, "wb") as f:
#                 f.write(response.content)
#         else:
#             print("Could not find favicon URL")
#     except (requests.exceptions.SSLError, ssl.SSLEOFError) as e:
#         print(f"Error fetching favicon for {url}: {e}")
#         print("Skipping this site.")
#     except requests.exceptions.ProxyError as e:
#         print(f"ProxyError occurred: {e}")
#         print("Skipping this site.")

# def get_favicon_url_from_html(soup, url):
#     favicon_url = None
#     for link in soup.find_all("link", {"rel": ["shortcut icon", "icon"]}):
#         favicon_url = link.get("href")
#         break
#     if favicon_url and not favicon_url.startswith("http"):
#         favicon_url = url + favicon_url

#     return favicon_url

# def download_favicons(links):
#     for i, link in enumerate(links, start=1):
#         ensure_http(link)
#         print(f"Iteration no: {i}, URL: {link}")
#         download_favicon(link)

def fetch_domain_data(domains):
    domain_data = {}
    for domain in domains:
        content = extract_website_content(domain)
        title = get_title(domain)
        domain_data[domain] = {'content': content, 'title': title}
        print(f"done for {domain}")
    return domain_data

@app.route('/', methods=['POST'])
def detect_phishing():
    file = request.files['file']
    child_domains = file.read().decode('utf-8').splitlines()
    
    parent_data = pd.read_csv(r"detect-phishing-domains-main\detect-phishing-domains-api\whitelist.csv")
    parent_domains = parent_data['domain'].values

    threshold_ratio = 0
    parent_child_dict = {}

    # Fetch content and title for all domains
    all_domain_data = fetch_domain_data(child_domains + list(parent_domains))

    # Separate child and parent domain data
    child_domain_data = {domain: all_domain_data[domain] for domain in child_domains}
    parent_domain_data = {domain: all_domain_data[domain] for domain in parent_domains}

    i = 1
    for parent in parent_domains:
        matching_children = []
        j = 1
        for child in child_domains:
            print(f"site {i} iteration {j}: {child}")
            j += 1
            ratio = fuzz.ratio(parent, child)
            if ratio >= threshold_ratio:
                try:
                    # Get content and title for parent domain from pre-fetched data
                    parent_content = parent_domain_data[parent]['content']
                    parent_title = parent_domain_data[parent]['title']
                    
                    # Retrieve content and title for child domain from pre-fetched data
                    child_content = child_domain_data[child]['content']
                    child_title = child_domain_data[child]['title']
                    
                    # Calculate domain similarity
                    domain_similarity = calculate_domain_similarity(parent, child)
                    
                    # Calculate text similarity
                    content_similarity = calculate_similarity(parent_content, child_content) if parent_content and child_content else 0.0
                    
                    # Compare titles
                    title_similarity = compare_titles(parent_title, child_title) if parent_title and child_title else 0.0

                    matching_children.append((child, {
                        'domain_similarity': domain_similarity,
                        'content_similarity': content_similarity,
                        'title_similarity': title_similarity,
                    }))
                except Exception as e:
                    error_message = f"Error processing {parent} and {child}: {e}"
                    matching_children.append((child, {'error': error_message}))
                    print(error_message)
        
        i += 1
        if matching_children:
            parent_child_dict[parent] = matching_children

    return jsonify(parent_child_dict)

def fetch_domain_data(domains):
    domain_data = {}
    for domain in domains:
        content = extract_website_content(domain)
        title = get_title(domain)
        domain_data[domain] = {'content': content, 'title': title}
    return domain_data

if __name__ == "__main__":
    app.run()
