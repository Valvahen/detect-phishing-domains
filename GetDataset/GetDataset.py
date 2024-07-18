import base64
import requests
import zipfile
import os
from datetime import datetime, timedelta
import csv

# Function to download and extract CSV from a dynamically generated URL
def download_and_extract_csv(start_date):
    # Convert the start date to the required string format
    str_start_date = start_date.strftime("%Y-%m-%d")
    print("Trying to download " + str_start_date + " data")
    
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
        zip_folder = "Datasets/ZIPS"
        extract_folder = "Datasets"
        
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
        
        # Rename the extracted domain-names.txt file to include the date
        original_file = os.path.join(extract_folder, "domain-names.txt")
        new_file = os.path.join(extract_folder, f"domain-names-{str_start_date}.txt")
        if os.path.exists(original_file):
            os.rename(original_file, new_file)
            print(f"Renamed {original_file} to {new_file}")

        print(f"Extracted {zip_filename} to {extract_folder}")

        # Log the date in a CSV file
        with open('downloaded_dates.csv', 'a', newline='') as csvfile:
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

# Main script
if __name__ == "__main__":
    csv_filename = 'downloaded_dates.csv'
    last_downloaded_date = get_last_downloaded_date(csv_filename)

    if last_downloaded_date:
        start_date = last_downloaded_date + timedelta(days=1)
    else:
        start_date = (datetime.now().date() - timedelta(days=3)) # Default start date if CSV is empty

    current_date = datetime.now().date()
    
    # Loop to download and extract all missing dates up to the current date
    next_date = start_date
    while next_date < current_date:
        download_and_extract_csv(next_date)
        next_date += timedelta(days=1)
