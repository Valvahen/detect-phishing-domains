import socket
from tqdm import tqdm
import concurrent.futures
import csv
from ..config import max_workers

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
def get_ip_addresses(domains):
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