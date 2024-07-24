from flask import jsonify, request, Flask, send_file
from flask_cors import CORS
import pandas as pd
import json
import time
from .modules.process_child_domain_in_batches import process_child_domains_in_batches
from .modules.determine_status import determine_status
from .modules.get_whois_dataset import get_whois_dataset
import os
from .config import whitelist
from .modules.filter_nixi_file import filter_nixi_file

# Flask setup
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def hello_world():
    return "Hello, World!"

@app.route('/detect-phishing', methods=['POST'])
def detect_phishing():
    selected_features = json.loads(request.form['features'])
    source = request.form['source']

    if source == 'whois':
        print("Processing with whois")
        base_filename = "my_app\\data\\results\\whoisresults.csv"
        parent_data = pd.read_csv(whitelist)
        parent_domains = parent_data['domain'].values
        
        # Fetch the WHOIS dataset file path
        file_path = get_whois_dataset()
        print(f"WHOIS dataset path: {file_path}")
        
        # Ensure that the file path is correctly read
        if not os.path.exists(file_path):
            return jsonify({"error": "WHOIS dataset file not found"}), 500
        
        # Read the content of the WHOIS dataset file
        with open(file_path, 'r') as file:
            child_domains = file.read().splitlines()
        
        start_processing_time = time.time()
        csv_filename = process_child_domains_in_batches(child_domains, parent_domains, selected_features, base_filename)
        end_processing_time = time.time()
        processing_time = end_processing_time - start_processing_time

        print(f"Time taken for processing: {processing_time} seconds")

        # Read the original CSV file
        df = pd.read_csv(csv_filename)

        # Check which columns exist in the original CSV
        existing_columns = df.columns.tolist()

        # Apply determine_status and append the status to the original DataFrame
        df['status'] = df.apply(determine_status, axis=1, existing_columns=existing_columns)

        # Save the DataFrame with the status column appended to the original CSV file
        df.to_csv(csv_filename, index=False)

        print("CSV file with status column saved successfully.")

        # Generate link to download CSV
        csv_download_link = f'http://127.0.0.1:5000/download/{csv_filename}'

        return jsonify({"csv_download_link": csv_download_link}), 200
        # Add your NIXI specific processing here
    elif source == 'nixi':
        
        print("Processing with nixi")
        # Add your WHOIS specific processing here

        file = request.files['file']
        child_domains = filter_nixi_file(file)
        # child_domains = file.read().decode('utf-8').splitlines()
        
        parent_data = pd.read_csv(whitelist)
        parent_domains = parent_data['domain'].values

        base_filename = f"my_app/data/results/{json.loads(request.form['date'])}.csv"


        start_processing_time = time.time()
        csv_filename = process_child_domains_in_batches(child_domains, parent_domains, selected_features, base_filename)
        end_processing_time = time.time()
        processing_time = end_processing_time - start_processing_time

        print(f"Time taken for processing: {processing_time} seconds")

        # Read the original CSV file
        df = pd.read_csv(csv_filename)

        # Check which columns exist in the original CSV
        existing_columns = df.columns.tolist()

        # Apply determine_status and append the status to the original DataFrame
        df['status'] = df.apply(determine_status, axis=1, existing_columns=existing_columns)

        # Save the DataFrame with the status column appended to the original CSV file
        df.to_csv(csv_filename, index=False)

        print("CSV file with status column saved successfully.")

        # Generate link to download CSV
        csv_download_link = f'http://127.0.0.1:5000/download/{csv_filename}'

        return jsonify({"csv_download_link": csv_download_link}), 200
    
@app.route('/download/<filename>', methods=['GET'])
def download_csv(filename):
    csv_path = os.path.join(app.root_path, filename)
    return send_file(csv_path, as_attachment=True)
  
if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    app.run(host='0.0.0.0', port=port)

