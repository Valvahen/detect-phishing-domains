import pandas as pd

def filter_nixi_file(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Filter the DataFrame to include only rows where 'Operation' is 'CREATE'
    filtered_domains_df = df[df['Operation'] == 'CREATE']
    
    # Extract the 'Domain User Form' column from the filtered DataFrame
    filtered_domains = filtered_domains_df['Domain User Form'].tolist()
    
    return filtered_domains
