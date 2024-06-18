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
from collections import defaultdict
import time
import csv

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
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            content = response.text
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find the main content of the webpage
            main_content = soup.find('main')  # You can adjust this according to the structure of the webpage
            
            if (main_content):
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

def calculate_similarity(paragraph1, paragraph2):
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
        return -1

def get_title(url):
    url = ensure_http(url)
    if url in title_cache:
        return title_cache[url]
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string.strip() if soup.title else 'No title found'
        title_cache[url] = title
        print(f"title fetched for {url}")
        return title
    except Exception as e:
        print(f"Error fetching title from {url}: {e}")
        return 'No title found'

def compare_titles(title1, title2):
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
        # Remove "www" prefix if present
        parent = parent.lower().replace("www.", "")
        child = child.lower().replace("www.", "")

        # Calculate positional Jaccard similarity
        parent_set = set(parent)
        child_set = set(child)
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
        content_futures = {executor.submit(extract_website_content, domain): domain for domain in domains}
        title_futures = {executor.submit(get_title, domain): domain for domain in domains}

        for content_future in concurrent.futures.as_completed(content_futures):
            domain = content_futures[content_future]
            try:
                content = content_future.result()
                domain_data[domain] = {'content': content}
            except Exception as e:
                print(f"Error fetching content for {domain}: {e}")

        for title_future in concurrent.futures.as_completed(title_futures):
            domain = title_futures[title_future]
            try:
                title = title_future.result()
                if domain in domain_data:
                    domain_data[domain]['title'] = title
                else:
                    domain_data[domain] = {'title': title}
            except Exception as e:
                print(f"Error fetching title for {domain}: {e}")

    return domain_data

def compare_domains(parent, child, parent_domain_data, child_domain_data, matching_children):
    try:
        parent_content = parent_domain_data[parent]['content']
        parent_title = parent_domain_data[parent]['title']
        child_content = child_domain_data[child]['content']
        child_title = child_domain_data[child]['title']

        domain_similarity = calculate_domain_similarity(parent, child)
        content_similarity = calculate_similarity(parent_content, child_content) if parent_content and child_content else 0.0
        title_similarity = compare_titles(parent_title, child_title) if parent_title and child_title else 0.0

        print(f"Comparing {parent} and {child}:")

        matching_children.append((child, {
            'domain_similarity': domain_similarity,
            'content_similarity': content_similarity,
            'title_similarity': title_similarity,
        }))
    except Exception as e:
        error_message = f"Error processing {parent} and {child}: {e}"
        matching_children.append((child, {'error': error_message}))
        print(error_message)

def save_results_to_csv(results, filename='results.csv'):
    # Convert the nested dictionary to a flat list of dictionaries
    flat_results = []
    for parent, children in results.items():
        for child, similarities in children:
            flat_result = {'parent_domain': parent, 'child_domain': child}
            flat_result.update(similarities)
            flat_results.append(flat_result)
    
    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(flat_results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

@app.route('/', methods=['POST'])
def detect_phishing():
    file = request.files['file']
    child_domains = file.read().decode('utf-8').splitlines()
    
    parent_data = pd.read_csv("detect-phishing-domains-main/detect-phishing-domains-api/whitelist.csv")
    parent_domains = parent_data['domain'].values

    threshold_ratio = 0
    parent_child_dict = defaultdict(list)
    start_scraping_time = time.time()
    
    # Fetch content and title for all domains
    all_domain_data = fetch_domain_data(child_domains + list(parent_domains))
    end_scraping_time = time.time()
    scraping_time = end_scraping_time - start_scraping_time

    # Separate child and parent domain data
    child_domain_data = {domain: all_domain_data[domain] for domain in child_domains}
    parent_domain_data = {domain: all_domain_data[domain] for domain in parent_domains}

    start_comparison_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
        futures = []
        for parent in parent_domains:
            for child in child_domains:
                ratio = fuzz.ratio(parent, child)
                if ratio >= threshold_ratio:
                    futures.append(executor.submit(compare_domains, parent, child, parent_domain_data, child_domain_data, parent_child_dict[parent]))
        
        # Wait for all futures to complete
        concurrent.futures.wait(futures)
            
    end_comparison_time = time.time()
    comparison_time = end_comparison_time - start_comparison_time

    # Save results to CSV file
    save_results_to_csv(parent_child_dict)

    print(f"Time taken for scraping: {scraping_time} seconds")
    print(f"Time taken for comparisons: {comparison_time} seconds")
    return jsonify(parent_child_dict)

if __name__ == "__main__":
    app.run()
