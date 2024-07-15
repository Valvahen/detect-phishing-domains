from flask import jsonify, request, Flask
from flask_cors import CORS
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import textdistance
import nltk
from nltk.corpus import stopwords
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
from ultralytics import YOLO
import cv2
from flask import send_file

# Suppress SSL warnings
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Download NLTK tokenizer data
# nltk.download('stopwords')
# nltk.download('punkt')

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

# Load the YOLO model
model = YOLO('best.pt')

# Directory to store screenshots
screenshot_dir = 'blacklist_screenshots_3'
os.makedirs(screenshot_dir, exist_ok=True)

# Set up Chrome options for headless mode
chromeOptions = webdriver.ChromeOptions()
chromeOptions.add_argument('--headless')
chromeOptions.add_argument('--disable-gpu')
chromeOptions.add_argument('--ignore-certificate-errors')
chromeOptions.add_argument("--log-level=1")

# Set up the Selenium WebDriver
driver = webdriver.Chrome(options=chromeOptions)

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

# Function to take screenshots
def take_screenshots_from_csv(csv_file):
    screenshots = []
    df = pd.read_csv(csv_file)
    domains = df['domain'].tolist()
    for domain in domains:
        try:
            url = ensure_http(domain)
            response = requests.options(url)
            if response.ok or response.status_code == 403:
                driver.get(url)
                driver.set_window_size(1920, 1080)  # Set window size for full page capture
                screenshot_path = os.path.join(screenshot_dir, f"{domain.replace('.', '_')}.png")
                driver.save_screenshot(screenshot_path)
                screenshots.append((url, screenshot_path))
                print(f"Screenshot saved to {screenshot_path}")
            else:
                print(f"{domain} is accessible but something is not right. Response code: {response.status_code}")
        except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
            print(f"Unable to establish connection: {e}.")
        except Exception as e:
            print(f"Error taking screenshot of {domain}: {e}")
            continue
    return screenshots

# Predefined list of objects
predefined_objects = {
    0: 'aadhar', 1: 'adani', 2: 'airtel', 3: 'assam_sldc', 4: 'axis', 5: 'bbmb',
    6: 'bescom', 7: 'bob', 8: 'boi', 9: 'bse', 10: 'bses', 11: 'bsnl', 12: 'canara',
    13: 'cdsl', 14: 'census_india', 15: 'central_bank_of_india', 16: 'cesc', 17: 'cptcl',
    18: 'csrorgi', 19: 'dgshipping', 20: 'dvc', 21: 'gail', 22: 'grid', 23: 'gst',
    24: 'hdfc', 25: 'hpsldc', 26: 'iccl', 27: 'icici', 28: 'idbi', 29: 'indian_bank',
    30: 'iocl', 31: 'isro', 32: 'jio', 33: 'kerala_sldc', 34: 'kotak', 35: 'kptcl',
    36: 'lic', 37: 'mccil', 38: 'mcx', 39: 'megsldc', 40: 'mpcz', 41: 'mpez', 42: 'mpsldc',
    43: 'mpwz', 44: 'mse', 45: 'mtnl', 46: 'nic', 47: 'npci', 48: 'npcl', 49: 'npl',
    50: 'nsdl', 51: 'nse', 52: 'paytm', 53: 'pnb', 54: 'power', 55: 'pstcl', 56: 'rajasthan_sldc',
    57: 'rbi', 58: 'sbi', 59: 'tamil_nadu_sldc', 60: 'tata_power', 61: 'tata_power_ddl',
    62: 'telangana_sldc', 63: 'uk_sldc', 64: 'union_bank', 65: 'videocon', 66: 'vodafone',
    67: 'wbsedcl', 68: 'yes_bank'
}

# Reverse the predefined_objects dictionary for easy lookup
predefined_labels = {v: k for k, v in predefined_objects.items()}

# Queue to manage URLs for Selenium content extraction
url_queue = Queue()

def extract_website_content(url):
    dom=url
    url = ensure_http(url)
    # Check cache first
    if url in content_cache:
        print(f"Content for {url} already in cache")
        return content_cache[url]
    
    
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
                # Cache the content
                content_cache[url] = content
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

def remove_stop_words(paragraph, lang='english'):
    stop_words = set(stopwords.words(lang))
    words = word_tokenize(paragraph)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

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
        
        # Remove stop words from the paragraphs
        paragraph1_filtered = remove_stop_words(paragraph1, lang=lang_paragraph1)
        paragraph2_filtered = remove_stop_words(paragraph2, lang=lang_paragraph2)
        
        # Concatenate paragraphs
        combined_paragraphs = [paragraph1_filtered, paragraph2_filtered]
        
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
    # Check cache first
    if url in title_cache:
        print(f"Title for {url} already in cache")
        return title_cache[url]
    
    
    try:
        # Make a GET request to the URL with allow_redirects=True to follow redirects
        response = requests.get(url, verify=False, allow_redirects=True, timeout=5)
        response.raise_for_status()  # Raise an exception for 4xx/5xx status codes
        
        # Get the final URL after following redirects
        final_url = response.url
        
        # Use BeautifulSoup to parse the content and find the title
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string.strip() if soup.title else 'No title found'
        
        # Cache the title
        title_cache[url] = title
        print(f"Title fetched for {url}")
        return title
    
    except requests.RequestException as e:
        print(f"Error fetching title from {url}: {e}")
        return 'No title found'

def compare_titles(title1, title2, n=2):
    if title1 == 'No title found' or title2 == 'No title found':
        return -1
    if title1 == 'Home' and title2 == 'Home':
        return 0
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

        # Ensure parent and child are strings
        parent = str(parent)
        child = str(child)

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

import os
import pandas as pd
import csv
import socket
from tqdm import tqdm
import math

def get_next_available_filename(base_filename):
    base_name, extension = os.path.splitext(base_filename)
    counter = 1
    new_filename = base_filename

    while os.path.exists(new_filename):
        new_filename = f"{base_name}{counter}{extension}"
        counter += 1

    return new_filename

# Function to read domains from a CSV file
def read_domains_from_csv(file_path):
    domains = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            domains.append(row['domain'])
    return domains

# Function to read known malicious IPs from a CSV file
def read_malicious_ips_from_csv(file_path):
    malicious_ips = set()
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            malicious_ips.add(row[0])
    return malicious_ips

# Function to get IP address for a single domain
def get_ip_address(domain):
    try:
        ip_address = socket.gethostbyname(domain)
        return domain, ip_address
    except socket.gaierror:
        return domain, 'Error: Could not resolve domain'

# Function to get IP addresses from a list of domains using threading
def get_ip_addresses(domains, max_workers=10):
    domain_ip_map = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map get_ip_address function to the domains with a progress bar
        results = list(tqdm(executor.map(get_ip_address, domains), total=len(domains), desc="Resolving IP addresses"))
    for domain, ip in results:
        domain_ip_map[domain] = ip
    return domain_ip_map

# Function to check if an IP address is malicious
def check_if_malicious(domain_ip_tuple, malicious_ips):
    domain, ip = domain_ip_tuple
    is_malicious = 1 if ip in malicious_ips else 0
    return domain, ip, is_malicious

# Function to save results to CSV
def save_results_to_csv(results, results_file, batch_index=None):
    flat_results = []
    for parent, children in results.items():
        for child, child_info in children:
            flat_result = {'parent_domain': parent, 'child_domain': child}
            flat_result.update(child_info)
            if batch_index is not None:
                flat_result['batch_index'] = batch_index  # Add batch index if provided
            flat_results.append(flat_result)

    if flat_results:  # Check if there are results to save
        df = pd.DataFrame(flat_results)
        if not os.path.exists(results_file):
            df.to_csv(results_file, mode='w', index=False)
        else:
            df.to_csv(results_file, mode='a', header=False, index=False)
        print(f"Batch {batch_index} results appended to {results_file}")
    else:
        print("No results to save.")

# Function to process child domains in batches
def process_child_domains_in_batches(child_domains, parent_domains, selected_features, base_filename='results.csv'):
    batch_size = 1000
    num_batches = math.ceil(len(child_domains) / batch_size)

    results_file = get_next_available_filename(base_filename) if os.path.exists(base_filename) else base_filename

    processed_child_domains = set()  # Track processed child domains

    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, len(child_domains))
        batch_child_domains = child_domains[start_index:end_index]

        results = {}  # Reset results for each batch

        # Fetch domain data for this batch (mockup function, replace with actual implementation)
        domain_data = fetch_domain_data(batch_child_domains + list(parent_domains), selected_features)

        # Separate child and parent domain data
        child_domain_data = {domain: domain_data.get(domain, {}) for domain in batch_child_domains}

        # Path to the known malicious IPs CSV file
        malicious_ips_file_path = 'known_malicious_IPs.csv'

        # Read known malicious IPs from the CSV file
        malicious_ips = read_malicious_ips_from_csv(malicious_ips_file_path)

        # Get IP addresses for the domains using threading
        domain_ip_map = get_ip_addresses(batch_child_domains, max_workers=10)

        # Check if IP addresses are malicious using threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Map check_if_malicious function to the domain_ip_map with a progress bar
            results_with_malicious_check = list(tqdm(executor.map(check_if_malicious, domain_ip_map.items(), [malicious_ips]*len(domain_ip_map)), total=len(domain_ip_map), desc="Checking for malicious IPs"))

        for parent in tqdm(parent_domains, desc=f"Processing batch {batch_index+1}/{num_batches}"):
            matching_children = []
            for child in tqdm(batch_child_domains, desc=f"Processing children for parent {parent}", leave=False):
                try:
                    result = {}

                    # Example feature calculations (replace with actual logic)
                    if 'domain' in selected_features:
                        domain_similarity = calculate_domain_similarity(parent, child)
                        result['domain_similarity'] = domain_similarity

                    if 'content' in selected_features:
                        parent_content = domain_data.get(parent, {}).get('content', '')
                        child_content = child_domain_data.get(child, {}).get('content', '')
                        content_similarity = calculate_similarity(parent_content, child_content) if parent_content and child_content else 0.0
                        result['content_similarity'] = content_similarity

                    if 'title' in selected_features:
                        parent_title = domain_data.get(parent, {}).get('title', '')
                        child_title = child_domain_data.get(child, {}).get('title', '')
                        title_similarity = compare_titles(parent_title, child_title) if parent_title and child_title else 0.0
                        result['title_similarity'] = title_similarity

                    if 'screenshot' in selected_features:
                        screenshot_similarity = compare_screenshots(parent, child)
                        result['screenshot_similarity'] = screenshot_similarity

                    # Add malicious IP check result to the result dictionary
                    for domain_ip_info in results_with_malicious_check:
                        if domain_ip_info[0] == child:
                            result['is_malicious'] = domain_ip_info[2]
                            break

                    # Find matching children
                    matching_children.append((child, result))

                    # Mark (parent, child) combination as processed
                    processed_child_domains.add((parent, child))

                except Exception as e:
                    error_message = f"Error processing {parent} and {child}: {e}"
                    matching_children.append((child, {'error': error_message}))
                    print(error_message)

            if matching_children:
                results[parent] = matching_children

        # Save results of the current batch to CSV file
        save_results_to_csv(results, results_file, batch_index=batch_index)

    return results_file

# Determine status based on conditions
def determine_status(row, existing_columns):
    status = 'safe'
    if 'domain_similarity' in existing_columns and 'title_similarity' in existing_columns and 'content_similarity' in existing_columns:
        if row['domain_similarity'] > 60 or row['title_similarity'] > 60 or row['content_similarity'] > 60:
            status = 'suspected'
        if (row['domain_similarity'] > 60 and row['title_similarity'] > 60) or (row['content_similarity'] > 60 and row['domain_similarity'] > 60) or (row['content_similarity'] > 60 and row['title_similarity'] > 60):
            status = 'phishing'
    elif 'domain_similarity' in existing_columns and 'title_similarity' in existing_columns:
        if row['domain_similarity'] > 60 or row['title_similarity'] > 60:
            status = 'suspected'
        if (row['domain_similarity'] > 60 and row['title_similarity'] > 60):
            status = 'phishing'
    elif 'domain_similarity' in existing_columns and 'content_similarity' in existing_columns:
        if row['domain_similarity'] > 60 or row['content_similarity'] > 60:
            status = 'suspected'
        if (row['domain_similarity'] > 60 and row['content_similarity'] > 60):
            status = 'phishing'
    elif 'title_similarity' in existing_columns and 'content_similarity' in existing_columns:
        if row['title_similarity'] > 60 or row['content_similarity'] > 60:
            status = 'suspected'
        if (row['title_similarity'] > 60 and row['content_similarity'] > 60):
            status = 'phishing'
    elif 'domain_similarity' in existing_columns and row['domain_similarity'] > 60:
        status = 'suspected'
    elif 'title_similarity' in existing_columns and row['title_similarity'] > 60:
        status = 'suspected'
    elif 'content_similarity' in existing_columns and row['content_similarity'] > 60:
        status = 'suspected'

    # Check if is_malicious is 1 and set status to phishing
    if 'is_malicious' in existing_columns and row.get('is_malicious') == 1:
        status = 'phishing'

    return status

@app.route('/', methods=['POST'])
def detect_phishing():
    file = request.files['file']
    child_domains = file.read().decode('utf-8').splitlines()
    
    parent_data = pd.read_csv(r"whitelists\whitelist4.csv")
    parent_domains = parent_data['domain'].values

    selected_features = json.loads(request.form['features'])
    
    start_scraping_time = time.time()
    csv_filename = process_child_domains_in_batches(child_domains, parent_domains, selected_features)
    end_scraping_time = time.time()
    scraping_time = end_scraping_time - start_scraping_time

    print(f"Time taken for scraping: {scraping_time} seconds")

    # Read the original CSV file
    df = pd.read_csv(csv_filename)

    # Check which columns exist in the original CSV
    existing_columns = df.columns.tolist()

    # Create a new DataFrame with parent_domain, child_domain, and status
    new_df = df[['parent_domain', 'child_domain']].copy()
    new_df['status'] = df.apply(determine_status, axis=1, existing_columns=existing_columns)

    # Filter out rows with status 'safe'
    new_df = new_df[new_df['status'] != 'safe']

    # Save the new DataFrame to a new CSV file
    processed_filename = 'processed.csv'
    new_df.to_csv(processed_filename, index=False)

    print("Processed CSV file saved successfully.")

    # Generate link to download CSV
    csv_download_link = f'http://127.0.0.1:5000/download/{csv_filename}'
    processed_file_download_link = f'http://127.0.0.1:5000/download/{processed_filename}'

    return jsonify({"csv_download_link": csv_download_link, "processed_file_download_link": processed_file_download_link}), 200

@app.route('/detect_logos', methods=['POST'])
def detect_logos():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        file_path = os.path.join(screenshot_dir, file.filename)
        file.save(file_path)

        # Take screenshots of domains from CSV
        screenshots = take_screenshots_from_csv(file_path)
        driver.quit()

        # List to store detection results
        results = []

        # Detect logos in screenshots
        for url, screenshot_path in screenshots:
            if os.path.exists(screenshot_path):
                # Example evaluation on a validation set
                results_yolo = model(screenshot_path)

                # Assuming results is an instance of ultralytics.engine.results.Results
                if isinstance(results_yolo, list) and len(results_yolo) > 0:
                    detected_objects = results_yolo[0].names
                    detected_labels = set()  # Store detected labels

                    # Load the image
                    img = cv2.imread(screenshot_path)

                    # Draw bounding boxes and labels on the image
                    for box in results_yolo[0].boxes:
                        # Convert tensor to list
                        box_coordinates = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = map(int, box_coordinates)
                        label = detected_objects[box.cls.item()]

                        # Check if the label is in the predefined objects
                        if label in predefined_labels:
                            detected_labels.add(label)

                            # Draw the bounding box
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            # Put the label text above the bounding box
                            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Save the image with bounding boxes (if needed)
                    cv2.imwrite(f"{screenshot_path[:-4]}_with_boxes.png", img)

                    # Add detected labels to results
                    for label in detected_labels:
                        results.append({'URL': url, 'Detected Logo': label})
                else:
                    print(f"Error: Unexpected results format for {screenshot_path}.")
            else:
                print(f"File not found: {screenshot_path}")

        # Save results to CSV
        df = pd.DataFrame(results)
        df.to_csv('detected_logos_3.csv', index=False)
        csv_filename = 'detected_logos_3.csv'
        csv_path = os.path.join(app.root_path, csv_filename)
        df.to_csv(csv_path, index=False)

        # Generate link to download CSV
        csv_download_link = f'http://127.0.0.1:5000/download/{csv_filename}'

        return jsonify({"csv_download_link": csv_download_link}), 200
    
@app.route('/download/<filename>', methods=['GET'])
def download_csv(filename):
    csv_path = os.path.join(app.root_path, filename)
    return send_file(csv_path, as_attachment=True)

if __name__ == "__main__":
    app.run()
