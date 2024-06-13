from bs4 import BeautifulSoup
from urllib.parse import urlparse
import requests
import os
from pandas import *

session = requests.Session()
session.verify = False
img_out_dir = r"detect-phishing-domains-main\detect-phishing-domains-api\favicons_wl"


def download_favicon(url, filename=None):
    parsed_url = urlparse(url)

    if not filename:
        # use second-level domain (SLD) for filename
        filename = parsed_url.netloc
    # check if favicon already exists
    favicon_output_filename = os.path.join(img_out_dir, filename + ".ico")
    if os.path.exists(favicon_output_filename):
        print(favicon_output_filename + " already exists!")
        return

    # get url without path
    url = parsed_url.scheme + "://" + parsed_url.netloc
    print(url)
    try:
        response = session.get(url)

        # parse and get the favicon URL from the HTML content
        soup = BeautifulSoup(response.content, "html.parser")
        favicon_url = get_favicon_url_from_html(soup, url)

        if favicon_url:
            # download the favicon
            response = session.get(favicon_url)
            with open(favicon_output_filename, "wb") as f:
                f.write(response.content)
        else:
            print("Could not find favicon URL")
    except Exception as e:
        print(f"Error fetching favicon for {url}: {e}")


def get_favicon_url_from_html(soup, url):
    favicon_url = None
    for link in soup.find_all("link", {"rel": ["shortcut icon", "icon"]}):
        favicon_url = link.get("href")
        break
    if favicon_url and not favicon_url.startswith("http"):
        favicon_url = url + favicon_url

    return favicon_url


def download_favicons(links):
    for link in links:
        download_favicon(link)


if __name__ == "__main__":
    urls = ["http://checkpethome.com/", "https://crsorgi.gov.in"]
    # data = read_csv(r"detect-phishing-domains-main\detect-phishing-domains-api\whitelist.csv")
    # urls = data['domain'].tolist()
    download_favicons(urls)
