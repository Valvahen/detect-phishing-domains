from .ensure_scheme import ensure_scheme
import requests
from bs4 import BeautifulSoup
import textdistance
from ..config import title_cache

def get_title(url):
    url = ensure_scheme(url)
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