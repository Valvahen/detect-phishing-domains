from flask import jsonify, request, Flask, send_file
from flask_cors import CORS
import pandas as pd
import json
import time
from .modules.process_child_domain_in_batches import process_child_domains_in_batches
from .modules.determine_status import determine_status
import os
import requests
import ssl
from .config import content_cache, title_cache, url_queue, whitelist, processed_filename

# Suppress SSL warnings
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Disable SSL certificate verification
session = requests.Session()
session.verify = False

# Flask setup
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def hello_world():
  print("Hello, World!")
  
@app.route('/detect-phishing', methods=['POST'])
def detect_phishing():
    file = request.files['file']
    child_domains = file.read().decode('utf-8').splitlines()
    
    parent_data = pd.read_csv(whitelist)
    parent_domains = parent_data['domain'].values

    selected_features = json.loads(request.form['features'])
    
    start_processing_time = time.time()
    csv_filename = process_child_domains_in_batches(child_domains, parent_domains, selected_features)
    end_processing_time = time.time()
    processing_time = end_processing_time - start_processing_time

    print(f"Time taken for processing: {processing_time} seconds")

    # Read the original CSV file
    df = pd.read_csv(csv_filename)

    # Check which columns exist in the original CSV
    existing_columns = df.columns.tolist()

    # Create a new DataFrame with parent_domain, child_domain, and status
    new_df = df[['Legitimate Domain', 'Newly Registered Domain']].copy()
    new_df['status'] = df.apply(determine_status, axis=1, existing_columns=existing_columns)

    # Filter out rows with status 'safe'
    new_df = new_df[new_df['status'] != 'safe']

    # Save the new DataFrame to a new CSV file
    new_df.to_csv(processed_filename, index=False)

    print("Processed CSV file saved successfully.")

    # Generate link to download CSV
    csv_download_link = f'http://127.0.0.1:5000/download/{csv_filename}'
    processed_file_download_link = f'http://127.0.0.1:5000/download/{processed_filename}'

    return jsonify({"csv_download_link": csv_download_link, "processed_file_download_link": processed_file_download_link}), 200
  
@app.route('/download/<filename>', methods=['GET'])
def download_csv(filename):
    csv_path = os.path.join(app.root_path, filename)
    return send_file(csv_path, as_attachment=True)
  
if __name__ == "__main__":
    app.run()