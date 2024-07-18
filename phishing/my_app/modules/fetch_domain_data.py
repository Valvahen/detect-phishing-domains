import concurrent.futures
from .content_similarity import extract_website_content, extract_website_content_using_selenium
from .title_similarity import get_title
from ..config import url_queue

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