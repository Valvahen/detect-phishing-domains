from flask import jsonify, request, Flask
from flask_cors import CORS
import pandas as pd
from thefuzz import fuzz
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import textdistance
import nltk
import concurrent.futures
import ssl

# Suppress SSL warnings
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Download NLTK tokenizer data
nltk.download('punkt')

# Disable SSL certificate verification
session = requests.Session()
session.verify = False

app = Flask(__name__)
CORS(app)

# Define dictionaries to store cached data
content_cache = {}
title_cache = {}

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
    
def fetch_domain_data(domains):
    domain_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_domain = {executor.submit(extract_website_content, domain): domain for domain in domains}
        for future in concurrent.futures.as_completed(future_to_domain):
            domain = future_to_domain[future]
            try:
                content = future.result()
                title = get_title(domain)
                domain_data[domain] = {'content': content, 'title': title}
            except Exception as e:
                print(f"Error fetching data for {domain}: {e}")
    return domain_data

@app.route('/', methods=['POST'])
def detect_phishing():
    file = request.files['file']
    child_domains = file.read().decode('utf-8').splitlines()
    
    parent_data = pd.read_csv(r".\whitelist.csv")
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

if __name__ == "__main__":
    app.run()
