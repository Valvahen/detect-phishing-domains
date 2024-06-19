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
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager
import os

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
        https_url = 'https://' + url
        try:
            requests.get(https_url)
            return https_url
        except requests.RequestException:
            http_url = 'http://' + url
            try:
                requests.get(http_url)
                return http_url
            except requests.RequestException:
                return url
    return url

def extract_text_with_selenium(url):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    service = Service(r"C:\Users\Intern\Downloads\geckodriver-v0.34.0-win64\geckodriver.exe") 
    
    driver = webdriver.Firefox(service=service, options=options)
    try:
        driver.get(url)
        time.sleep(5)  # Wait for JavaScript to render
        text = driver.find_element(By.TAG_NAME, 'body').text
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        text = ''
    finally:
        driver.quit()
    return text

def extract_website_content(url):
    url = ensure_http(url)
    if url in content_cache:
        return content_cache[url]
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            content = response.text
            if "You need to enable JavaScript to run this app." in content:
                print(f"JavaScript required for {url}, fetching content using Selenium...")
                content = extract_text_with_selenium(url)
            else:
                soup = BeautifulSoup(content, 'html.parser')
                main_content = soup.find('main')
                if main_content:
                    content = main_content.get_text(separator='\n')
                else:
                    if soup.body:
                        content = soup.body.get_text(separator='\n')
                    else:
                        content = ''
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
    text = ' '.join(text.split())
    return text

from langdetect import detect

def calculate_similarity(paragraph1, paragraph2):
    if paragraph1 == "No content found" or paragraph2 == "No content found":
        return -1
    try:
        combined_paragraphs = [paragraph1, paragraph2]
        tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenize, lowercase=True, norm=None)
        tfidf_matrix = tfidf_vectorizer.fit_transform(combined_paragraphs)
        tfidf_paragraph1 = tfidf_matrix[0]
        tfidf_paragraph2 = tfidf_matrix[1]
        similarity = cosine_similarity(tfidf_paragraph1, tfidf_paragraph2)[0][0]
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
        similarity_score = textdistance.damerau_levenshtein.normalized_similarity(title1.lower(), title2.lower())
        similarity_percentage = similarity_score * 100
        return similarity_percentage
    except Exception as e:
        print(f"Error comparing titles: {e}")
        return -1

# Function to strip TLD from domain, including multi-part TLDs like ".co.in"
def strip_tld(domain):
    multi_part_tlds = ['.co.in', '.gov.in', 'org.in']
    for tld in multi_part_tlds:
        if domain.endswith(tld):
            return domain[:-len(tld)]
    parts = domain.split('.')
    if len(parts) > 1:
        return '.'.join(parts[:-1])
    return domain

# Function to remove specified substrings from the domain
def remove_substrings(domain, substrings):
    for substring in substrings:
        domain = domain.replace(substring, "")
    return domain

# Load whitelist data from CSV
whitelist_data = {}
with open(r'detect-phishing-domains-main\detect-phishing-domains-api\whitelistSBI.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        whitelist_data[rows[0]] = rows[1]

def calculate_domain_similarity(parent, child):
    if not parent or not child:
        return -1
    try:
        # Define substrings to be removed
        substrings_to_remove = [
            "xyz", "abc", "123", "online", "site", "shop", "store", "web", "info",
            "net", "my", "the", "best", "top", "pro", "plus", "go", "free", "biz",
            "crt", "krt", 'india'
        ]

        # Remove "www" prefix if present
        parent = parent.lower().replace("www.", "")
        child = child.lower().replace("www.", "")

        # Check if 'gov' exists in both parent and child domains before removing substrings or stripping TLD
        similarity_increase = 0
        if "gov" in parent and "gov" in child:
            similarity_increase += 5

        # Remove specified substrings
        parent = remove_substrings(parent, substrings_to_remove)
        child = remove_substrings(child, substrings_to_remove)

        # Calculate Damerau-Levenshtein similarity with TLD
        levenshtein_similarity_with_TLD = textdistance.damerau_levenshtein.normalized_similarity(parent, child) * 100

        # Calculate Damerau-Levenshtein similarity without TLD
        parent_stripped = strip_tld(parent)
        child_stripped = strip_tld(child)
        levenshtein_similarity_without_TLD = textdistance.damerau_levenshtein.normalized_similarity(parent_stripped, child_stripped) * 100

        # Calculate additional similarity metrics only if the stripped Levenshtein similarity is below a threshold
        jaccard_similarity_with_TLD = 0
        if levenshtein_similarity_with_TLD < 100:
            # Calculate positional Jaccard similarity with TLD
            parent_set = set(parent)
            child_set = set(child)
            intersection_count = len(parent_set.intersection(child_set))
            union_count = len(parent_set.union(child_set))
            jaccard_similarity_with_TLD = (intersection_count / union_count) * 100

        jaccard_similarity_without_TLD = 0
        if levenshtein_similarity_with_TLD < 100:
            # Calculate positional Jaccard similarity without TLD
            parent_without_tld_set = set(parent_stripped)
            child_without_tld_set = set(child_stripped)
            intersection_count = len(parent_without_tld_set.intersection(child_without_tld_set))
            union_count = len(parent_without_tld_set.union(child_without_tld_set))
            jaccard_similarity_without_TLD = (intersection_count / union_count) * 100

        # Check if 'org' exists in either domain names after stripping substrings
        parent_org = whitelist_data.get(parent, '')

        if parent_org and (parent_org in child or parent_org in parent):
            similarity_increase += 25

        # Use a weighted average to combine the similarities
        combined_similarity = (0.60 * levenshtein_similarity_without_TLD + 0.25 * jaccard_similarity_without_TLD + 0.10 * levenshtein_similarity_with_TLD + 0.05 * jaccard_similarity_with_TLD) + similarity_increase

        return min(combined_similarity, 100)  # Ensure similarity does not exceed 100
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
                if domain not in domain_data:
                    domain_data[domain] = {'title': 'No title found', 'content': 'No content found'}

    return domain_data

def compare_domains(parent, child, parent_domain_data, child_domain_data, matching_children):
    try:
        parent_content = parent_domain_data[parent].get('content', 'No content found')
        parent_title = parent_domain_data[parent].get('title', 'No title found')
        child_content = child_domain_data[child].get('content', 'No content found')
        child_title = child_domain_data[child].get('title', 'No title found')

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

def save_results_to_csv(results, results_folder='results', filename_base='results'):
    # Ensure the results folder exists
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Find existing files to determine the next available filename
    existing_files = [filename for filename in os.listdir(results_folder) if filename.startswith(filename_base)]
    if existing_files:
        last_file_number = max([int(filename.replace(filename_base, '').replace('.csv', '')) for filename in existing_files])
        next_file_number = last_file_number + 1
    else:
        next_file_number = 1

    filename = f"{filename_base}{next_file_number}.csv"
    filepath = os.path.join(results_folder, filename)

    flat_results = []
    for parent, children in results.items():
        for child, similarities in children:
            flat_result = {'parent_domain': parent, 'child_domain': child}
            flat_result.update(similarities)
            flat_results.append(flat_result)
    df = pd.DataFrame(flat_results)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")

@app.route('/', methods=['POST'])
def detect_phishing():
    file = request.files['file']
    child_domains = file.read().decode('utf-8').splitlines()
    
    parent_data = pd.read_csv("detect-phishing-domains-main/detect-phishing-domains-api/whitelistSBI.csv")
    parent_domains = parent_data['domain'].values

    threshold_ratio = 0
    parent_child_dict = defaultdict(list)
    start_scraping_time = time.time()
    
    all_domain_data = fetch_domain_data(child_domains + list(parent_domains))
    end_scraping_time = time.time()
    scraping_time = end_scraping_time - start_scraping_time

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
        
        concurrent.futures.wait(futures)
            
    end_comparison_time = time.time()
    comparison_time = end_comparison_time - start_comparison_time

    save_results_to_csv(parent_child_dict)

    print(f"Time taken for scraping: {scraping_time} seconds")
    print(f"Time taken for comparisons: {comparison_time} seconds")
    return jsonify(parent_child_dict)

if __name__ == "__main__":
    app.run()
