import pandas as pd
import os

# Function to save results to CSV
def save_results_to_csv(results, results_file, batch_index=None, domain_sector_map=None):
    flat_results = []
    for parent, children in results.items():
        for child, child_info in children:
            flat_result = {'Legitimate Domain': parent, 'Newly Registered Domain': child}
            flat_result.update(child_info)
            if batch_index is not None:
                flat_result['batch_index'] = batch_index  # Add batch index if provided
            if domain_sector_map and parent in domain_sector_map:
                flat_result['sector'] = domain_sector_map[parent]  # Add sector information
            flat_results.append(flat_result)

    if flat_results:  # Check if there are results to save
        df = pd.DataFrame(flat_results)
        if not os.path.exists(results_file):
            df.to_csv(results_file, mode='w', index=False)
        else:
            df.to_csv(results_file, mode='a', header=False, index=False)
        print(f"Batch {batch_index} results appended to {results_file}")
    else:
        print("No results to save.")