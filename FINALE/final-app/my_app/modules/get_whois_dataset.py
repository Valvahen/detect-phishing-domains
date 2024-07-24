import requests
import os
import base64
import zipfile
from datetime import datetime, timedelta
import csv
from ..config import csv_filename, whois_extracted_dateset_directory_path, whois_zips_dateset_directory_path

# Function to download and extract CSV from a dynamically generated URL
def download_and_extract_csv(start_date, combined_csv_writer):
    # Convert the start date to the required string format
    str_start_date = start_date.strftime("%Y-%m-%d")
    
    # Encode the date to base64
    date_zip = str_start_date + ".zip"
    random_str = base64.b64encode(date_zip.encode("utf-8")).decode("utf-8")
    
    # Construct the URL
    url = "https://www.whoisds.com/whois-database/newly-registered-domains/" + random_str + "/nrd"
    
    # Make the request to download the ZIP file
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Define paths
        zip_folder = whois_zips_dateset_directory_path
        extract_folder = whois_extracted_dateset_directory_path
        
        # Ensure the ZIP folder exists
        os.makedirs(zip_folder, exist_ok=True)
        
        # Save the ZIP file in the ZIPS folder
        zip_filename = os.path.join(zip_folder, date_zip)
        with open(zip_filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {zip_filename}")

        # Extract the CSV file from the ZIP into the Datasets folder
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)  # Extracts all files in the ZIP to the Datasets directory
        
        # Process and append data to combined.csv
        original_file = os.path.join(extract_folder, "domain-names.txt")
        if os.path.exists(original_file):
            with open(original_file, 'r') as infile:
                for line in infile:
                    combined_csv_writer.writerow([line.strip()])
        
        # Rename the extracted domain-names.txt file to include the date
        new_file = os.path.join(extract_folder, f"domain-names-{str_start_date}.txt")
        os.rename(original_file, new_file)

        # Log the date in a CSV file
        with open(csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([str_start_date])
        print(f"Logged {str_start_date} to downloaded_dates.csv")

    else:
        print(f"Failed to download data for {str_start_date}. Status code: {response.status_code}")

# Function to get the last downloaded date from the CSV file
def get_last_downloaded_date(csv_filename):
    if os.path.exists(csv_filename):
        with open(csv_filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            dates = list(csv_reader)
            if dates and dates[-1]:  # Check if dates list is not empty and the last entry is not empty
                last_date_str = dates[-1][0]
                last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
                return last_date
    return None

def get_whois_dataset():
    last_downloaded_date = get_last_downloaded_date(csv_filename)
    if last_downloaded_date:
        start_date = last_downloaded_date + timedelta(days=1)
    else:
        start_date = (datetime.now().date() - timedelta(days=3)) # Default start date if CSV is empty

    next_date = start_date
    current_date = datetime.now().date()
    
    # Path for combined CSV
    combined_csv_path = os.path.join(whois_extracted_dateset_directory_path, 'combined.csv')
    
    # Open combined CSV for writing (create if it doesn't exist, otherwise append)
    file_mode = 'a' if os.path.exists(combined_csv_path) else 'w'
    
    with open(combined_csv_path, file_mode, newline='') as combined_csv_file:
        combined_csv_writer = csv.writer(combined_csv_file)
        
        any_data_downloaded = False
        
        while next_date < current_date:
            download_and_extract_csv(next_date, combined_csv_writer)
            if os.path.getsize(combined_csv_path) > 0:
                any_data_downloaded = True
            next_date += timedelta(days=1)
        
        if not any_data_downloaded:
            print("No new data downloaded. Combined CSV file is unchanged.")

    print("Successfully handled WHOIS dataset")
    
    # Ensure the format matches the NIXI data
    # Assuming WHOIS data has a single column of domains
    whois_csv_path = combined_csv_path  # This path should be consistent with what NIXI provides
    
    # You might need to rename or reformat the CSV file if necessary
    return whois_csv_path
