from .ensure_scheme import ensure_scheme
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
from langdetect import detect
import time
from ..config import content_cache, url_queue

def extract_website_content(url):
    dom=url
    url = ensure_scheme(url)
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
        # print(f"Error fetching content from {url}: {e}")
        return "No content found"

def extract_website_content_using_selenium(url):
    url = ensure_scheme(url)
    
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
        # print(f"Error fetching content from {url} using Selenium: {e}")
        return "No content found"

def clean_text(text):
    # Remove extra spaces and newlines
    text = ' '.join(text.split())
    return text

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
        # print(f"Error calculating similarity: {e}")
        return -1