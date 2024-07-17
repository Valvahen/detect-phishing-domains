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

whitelist = "my_app\data\whitelist.csv"