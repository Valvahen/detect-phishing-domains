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
import json
import random
import os
from queue import Queue
import threading
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
import time

# Suppress SSL warnings
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Download NLTK tokenizer data
nltk.download('punkt')

# Disable SSL certificate verification
session = requests.Session()
session.verify = False

# Flask setup
app = Flask(__name__)
CORS(app)

# Define dictionaries to store cached data
content_cache = {}
title_cache = {}

# Semaphore to limit Selenium driver instances
driver_semaphore = threading.Semaphore(1)

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

# Queue to manage URLs for Selenium content extraction
url_queue = Queue()

def extract_website_content(url):
    dom=url
    url = ensure_http(url)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    try:
        # Try with HEAD request first
        response = requests.head(url, headers=headers, verify=True, timeout=10)
        if response.status_code != 200:
            # If HEAD request fails, try GET request directly
            response = requests.get(url, headers=headers, timeout=10, verify=True)
        else:
            # If HEAD request is successful, follow up with GET request
            response = requests.get(url, headers=headers, timeout=10, verify=True)
        
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

            # Check if the content suggests JavaScript is needed
            if "You need to enable JavaScript to run this app." in content.strip():
                print(f"Enqueueing {dom} for Selenium processing")
                url_queue.put(dom)  # Enqueue URL for Selenium processing
                return "No content found"  # Return placeholder since content will be fetched by Selenium
            else:    
                print(f"Content fetched for {url}")
                return content.strip()  # Clean up whitespace and return content

        else:
            print(f"Failed to fetch content from {url}. Status code: {response.status_code}")
            return "No content found"
    
    except requests.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return "No content found"

def extract_website_content_using_selenium(url):
    url = ensure_http(url)
    
    # Set up Selenium with headless Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Path to the ChromeDriver
    driver_path = r"chromedriver-win64/chromedriver.exe"
    service = Service(driver_path)
    
    try:
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(url)
        
        # Wait for JavaScript to load the content (adjust as needed)
        time.sleep(5)
        
        content = driver.page_source
        driver.quit()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Find the main content of the webpage
        main_content = soup.find('main')  # Adjust this according to the structure of the webpage
        
        if main_content:
            content = main_content.get_text(separator='\n')
        else:
            # Check if body tag exists
            if soup.body:
                content = soup.body.get_text(separator='\n')
            else:
                # If neither <main> nor <body> tag exists, return an empty string
                content = ''

        print(f"Content fetched for {url}")     
        return content
    
    except WebDriverException as e:
        print(f"Error fetching content from {url} using Selenium: {e}")
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
    try:
        # Make a GET request to the URL with allow_redirects=True to follow redirects
        response = requests.get(url, verify=False, allow_redirects=True, timeout=5)
        response.raise_for_status()  # Raise an exception for 4xx/5xx status codes
        
        # Get the final URL after following redirects
        final_url = response.url
        
        # Use BeautifulSoup to parse the content and find the title
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string.strip() if soup.title else 'No title found'
        
        print(f"Title fetched for {url}: {title}")
        return title
    
    except requests.RequestException as e:
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

# Function to strip TLD from domain, including multi-part TLDs like ".co.in"
def strip_tld(domain):
    multi_part_tlds = ['.co.in', '.org.in', '..in', '.in']
    for tld in multi_part_tlds:
        if domain.endswith(tld):
            return domain[:-len(tld)]
    parts = domain.split('.')
    if len(parts) > 2:
        return '.'.join(parts[:-1])
    return domain

# Function to remove specified substrings from the domain
def remove_substrings(domain, substrings):
    for substring in substrings:
        domain = domain.replace(substring, "")
    return domain

def calculate_domain_similarity(parent, child):
    if not parent or not child:
        return -1
    try:
        # Define substrings to be removed
        substrings_to_remove = [
            "xyz", "abc", "123", "online", "site", "shop", "store", "web", "info",
            "net", "my", "the", "best", "top", "pro", "plus", "gov", "free", "biz",
            "crt", "krt", 'india', 'mart', 'bank', 'customer', 'service', 'www.','credit'
        ]

        # Remove specified substrings
        parent_cleaned = remove_substrings(parent, substrings_to_remove)
        child_cleaned = remove_substrings(child, substrings_to_remove)

        # Calculate Damerau-Levenshtein similarity with TLD
        levenshtein_similarity_with_TLD = textdistance.damerau_levenshtein.normalized_similarity(parent_cleaned, child_cleaned) * 100

        # Calculate Damerau-Levenshtein similarity without TLD
        parent_stripped = strip_tld(parent_cleaned)
        child_stripped = strip_tld(child_cleaned)
        levenshtein_similarity_without_TLD = textdistance.damerau_levenshtein.normalized_similarity(parent_stripped, child_stripped) * 100

        # Calculate additional similarity metrics only if the stripped Levenshtein similarity is below a threshold
        jaccard_similarity_with_TLD = 0
        # Calculate positional Jaccard similarity with TLD
        parent_set = set(parent_cleaned)
        child_set = set(child_cleaned)
        intersection_count = len(parent_set.intersection(child_set))
        union_count = len(parent_set.union(child_set))
        jaccard_similarity_with_TLD = (intersection_count / union_count) * 100

        jaccard_similarity_without_TLD = 0

        # Calculate positional Jaccard similarity without TLD
        parent_without_tld_set = set(parent_stripped)
        child_without_tld_set = set(child_stripped)
        intersection_count = len(parent_without_tld_set.intersection(child_without_tld_set))
        union_count = len(parent_without_tld_set.union(child_without_tld_set))
        jaccard_similarity_without_TLD = (intersection_count / union_count) * 100

        # Use a weighted average to combine the similarities
        combined_similarity = (0.35 * levenshtein_similarity_without_TLD + 0.30 * jaccard_similarity_without_TLD + 0.05 * levenshtein_similarity_with_TLD + 0.30 * jaccard_similarity_with_TLD)

        return min(combined_similarity, 100)  # Ensure similarity does not exceed 100
    except Exception as e:
        print(f"Error calculating domain similarity: {e}")
        return -1

def fetch_domain_data(domains, features):
    domain_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        if 'content' in features:
            content_futures = {executor.submit(extract_website_content, domain): domain for domain in domains}
        if 'title' in features:
            title_futures = {executor.submit(get_title, domain): domain for domain in domains}

        if 'content' in features:
            for content_future in concurrent.futures.as_completed(content_futures):
                domain = content_futures[content_future]
                try:
                    content = content_future.result()
                    if domain in domain_data:
                        domain_data[domain]['content'] = content
                    else:
                        domain_data[domain] = {'content': content}
                except Exception as e:
                    print(f"Error fetching content for {domain}: {e}")

        if 'title' in features:
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

    # Process URLs in the queue that require Selenium for content extraction
    while not url_queue.empty():
        domain = url_queue.get()
        content = extract_website_content_using_selenium(domain)
        if domain in domain_data:
            domain_data[domain]['content'] = content
        else:
            domain_data[domain] = {'content': content}
        url_queue.task_done()

    return domain_data

# Placeholder function for screenshot similarity
def compare_screenshots(url1, url2):
    if not url1 or not url2:
        return -1
    try:
        # Simulate screenshot similarity with random value
        similarity_percentage = random.uniform(0, 100)
        return similarity_percentage
    except Exception as e:
        print(f"Error comparing screenshots: {e}")
        return -1

def save_results_to_csv(results, results_folder='results', filename_base='results'):
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    flat_results = []
    for parent, children in results.items():
        for child, child_info in children:
            flat_result = {'parent_domain': parent, 'child_domain': child}
            flat_result.update(child_info)
            flat_results.append(flat_result)

    if flat_results:  # Check if there are results to save
        base_filepath = os.path.join(results_folder, filename_base)
        filepath = base_filepath + ".csv"
        file_index = 1

        while os.path.exists(filepath):
            filepath = f"{base_filepath}{file_index}.csv"
            file_index += 1

        df = pd.DataFrame(flat_results)
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
    else:
        print("No results to save.")

@app.route('/', methods=['POST'])
def detect_phishing():
    file = request.files['file']
    child_domains = file.read().decode('utf-8').splitlines()
    
    parent_data = pd.read_csv(r"whitelist.csv")
    parent_domains = parent_data['domain'].values

    selected_features = json.loads(request.form['features'])
    
    threshold_ratio = 0
    parent_child_dict = {}
    
    start_scraping_time = time.time()
    domain_data = fetch_domain_data(child_domains + list(parent_domains), selected_features)
    end_scraping_time = time.time()
    scraping_time = end_scraping_time - start_scraping_time

    # Separate child and parent domain data
    child_domain_data = {domain: domain_data.get(domain, {}) for domain in child_domains}
    parent_domain_data = {domain: domain_data.get(domain, {}) for domain in parent_domains}

    start_comparison_time = time.time()

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
                    result = {}
                    if 'domain' in selected_features:
                        # Calculate domain similarity
                        domain_similarity = calculate_domain_similarity(parent, child)
                        result['domain_similarity'] = domain_similarity
                    
                    if 'content' in selected_features:
                        # Get content for parent and child domains from pre-fetched data
                        parent_content = parent_domain_data[parent].get('content', '')
                        child_content = child_domain_data[child].get('content', '')
                        
                        # Calculate text similarity
                        content_similarity = calculate_similarity(parent_content, child_content) if parent_content and child_content else 0.0
                        result['content_similarity'] = content_similarity
                    
                    if 'title' in selected_features:
                        # Get titles for parent and child domains from pre-fetched data
                        parent_title = parent_domain_data[parent].get('title', '')
                        child_title = child_domain_data[child].get('title', '')
                        
                        # Compare titles
                        title_similarity = compare_titles(parent_title, child_title) if parent_title and child_title else 0.0
                        result['title_similarity'] = title_similarity

                    if 'screenshot' in selected_features:
                        # Compare screenshots using placeholder function
                        screenshot_similarity = compare_screenshots(parent, child)
                        result['screenshot_similarity'] = screenshot_similarity

                    matching_children.append((child, result))
                except Exception as e:
                    error_message = f"Error processing {parent} and {child}: {e}"
                    matching_children.append((child, {'error': error_message}))
                    print(error_message)
        
        i += 1
        if matching_children:
            parent_child_dict[parent] = matching_children
    
    end_comparison_time = time.time()
    comparison_time = end_comparison_time - start_comparison_time

    save_results_to_csv(parent_child_dict)

    print(f"Time taken for scraping: {scraping_time} seconds")
    print(f"Time taken for comparisons: {comparison_time} seconds")
    return jsonify(parent_child_dict)

if __name__ == "__main__":
    app.run()
