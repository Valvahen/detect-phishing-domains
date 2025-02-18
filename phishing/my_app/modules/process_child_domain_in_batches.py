from .get_next_available_filename import get_next_available_filename
from .fetch_domain_data import fetch_domain_data
from .malicious_ip import read_malicious_ips_from_csv, get_ip_addresses, check_if_malicious
from .redirecting import fetch_urls_and_check_redirect
from .domain_similarity import calculate_domain_similarity
from .content_similarity import calculate_content_similarity
from .title_similarity import compare_titles
from .save_results_to_csv import save_results_to_csv
import concurrent.futures
import math
import os
from tqdm import tqdm
from ..config import malicious_ips_file_path

# Update the process_child_domains_in_batches function
def process_child_domains_in_batches(child_domains, parent_domains, selected_features, base_filename='my_app/data/results/results.csv'):
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

        if 'ipaddress' in selected_features:
            # Read known malicious IPs from the CSV file
            malicious_ips = read_malicious_ips_from_csv(malicious_ips_file_path)

            # Get IP addresses for the domains using threading
            domain_ip_map = get_ip_addresses(batch_child_domains, max_workers=100)

            # Check if IP addresses are malicious using threading
            with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
                # Map check_if_malicious function to the domain_ip_map with a progress bar
                results_with_malicious_check = list(tqdm(executor.map(check_if_malicious, domain_ip_map.items(), [malicious_ips]*len(domain_ip_map)), total=len(domain_ip_map), desc="Checking for malicious IPs"))

        if 'redirection' in selected_features:
            # Fetch URLs and check redirects
            url_results = fetch_urls_and_check_redirect(batch_child_domains, parent_domains)

        for parent in tqdm(parent_domains, desc=f"Processing batch {batch_index+1}/{num_batches}"):
            matching_children = []
            for child in tqdm(batch_child_domains, desc=f"Processing children for parent {parent}", leave=False):
                try:
                    result = {}

                    # Example feature calculations (replace with actual logic)
                    if 'domain' in selected_features:
                        domain_similarity = calculate_domain_similarity(parent, child)
                        result['Domain Similarity'] = domain_similarity

                    if 'content' in selected_features:
                        parent_content = domain_data.get(parent, {}).get('content', '')
                        child_content = child_domain_data.get(child, {}).get('content', '')
                        content_similarity = calculate_content_similarity(parent_content, child_content) if parent_content and child_content else 0.0
                        result['Content Similarity'] = content_similarity

                    if 'title' in selected_features:
                        parent_title = domain_data.get(parent, {}).get('title', '')
                        child_title = child_domain_data.get(child, {}).get('title', '')
                        title_similarity = compare_titles(parent_title, child_title) if parent_title and child_title else 0.0
                        result['Title Similarity'] = title_similarity

                    # Add malicious IP check result to the result dictionary
                    if 'ipaddress' in selected_features:
                        for domain_ip_info in results_with_malicious_check:
                            if domain_ip_info[0] == child:
                                result['Malicious IP'] = domain_ip_info[2]
                                break

                    # Add URL redirect check result to the result dictionary
                    if 'redirection' in selected_features:
                        for url_result in url_results:
                            if url_result[0] == child:
                                result['Redirects to Legitimate Domain'] = url_result[1]
                            else:
                                result['Redirects to Legitimate Domain'] = url_result[1]

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