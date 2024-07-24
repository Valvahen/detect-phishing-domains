import nltk
from queue import Queue
import requests
import ssl

# Suppress SSL warnings
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Disable SSL certificate verification
session = requests.Session()
session.verify = False

# Download NLTK tokenizer data
nltk.download('stopwords')
nltk.download('punkt')

# Define dictionaries to store cached data
content_cache = {}
title_cache = {}

# Queue to manage URLs for Selenium content extraction
url_queue = Queue()

whitelist = "my_app\\data\\whitelist.csv"
processed_filename = 'my_app\\data\\processed.csv'
malicious_ips_file_path = 'my_app\\data\\known_malicious_IPs.csv'
blacklist = "my_app\\data\\blacklist.csv"
csv_filename = "my_app\\data\\downloaded_dates.csv"
whois_extracted_dateset_directory_path = "my_app\\data\\datasets\\extracted"
whois_zips_dateset_directory_path = "my_app\\data\\datasets\\zips"

max_workers = 100