import ssl
import socket
from cryptography import x509
from cryptography.hazmat.backends import default_backend
import idna
import pandas as pd


def get_certificate(hostname, port=443):
    try:
        # Encode the hostname in IDNA (punycode) format
        hostname_idna = idna.encode(hostname)

        # Connect to the server
        context = ssl.create_default_context()
        conn = context.wrap_socket(socket.socket(socket.AF_INET), server_hostname=hostname_idna)
        conn.settimeout(5.0)
        conn.connect((hostname_idna, port))

        # Get the certificate in DER format
        der_cert = conn.getpeercert(True)
        conn.close()

        # Parse the certificate
        cert = x509.load_der_x509_certificate(der_cert, default_backend())
        return cert
    except Exception as e:
        print(f"Failed to get certificate for {hostname}: {e}")
        return None


def extract_organization(cert):
    try:
        # Extract the organization (O) field from the subject
        subject = cert.subject
        organization = None
        for attribute in subject:
            if attribute.oid == x509.NameOID.ORGANIZATION_NAME:
                organization = attribute.value
                break
        return organization
    except Exception as e:
        print(f"Failed to extract organization: {e}")
        return None


def get_organizations_from_csv(csv_file):
    domains = pd.read_csv(csv_file)['domain']
    results = []

    for domain in domains:
        cert = get_certificate(domain)
        org = extract_organization(cert)
        results.append((domain, org))

    return results


def highlight_none_org(results):
    for domain, org in results:
        if org is None:
            print(f"Domain: {domain} - Organization: None")
        else:
            print(f"Domain: {domain} - Organization: {org}")


# Example usage
csv_file = 'domain-names.csv'
results = get_organizations_from_csv(csv_file)
highlight_none_org(results)
