from .ensure_scheme import ensure_scheme
from urllib.parse import urlparse
import requests
import concurrent.futures
from tqdm import tqdm
from requests.exceptions import RequestException, Timeout

# Function to extract domain from URL
def get_domain_from_url(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

# Function to fetch URLs and check redirects
def fetch_urls_and_check_redirect(urls, whitelist):
    results = []

    def fetch_url(url):
        try:
            url = ensure_scheme(url)
            response = requests.get(url, allow_redirects=True, verify=False)
            final_url = response.url
            final_domain = get_domain_from_url(final_url)
            if final_domain in whitelist:
                print(f"Yes {url}")
                return (url, 1)
            else:
                print(f"No {url}")
                return (url, 0)
        except Timeout:
            return (url, -1)
        except RequestException:
            return (url, -1)
        except Exception:
            return (url, -1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        future_to_url = {executor.submit(fetch_url, url): url for url in urls}
        for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(urls), desc="Fetching URLs"):
            url = future_to_url[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append((result))
    return results