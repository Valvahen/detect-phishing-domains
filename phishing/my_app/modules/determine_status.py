# Determine status based on conditions
def determine_status(row, existing_columns):
    status = 'safe'
    if 'Domain Similarity' in existing_columns and 'Title Similarity' in existing_columns and 'Content Similarity' in existing_columns:
        if row['Domain Similarity'] > 60 or row['Title Similarity'] > 60 or row['Content Similarity'] > 60:
            status = 'suspected'
        if (row['Domain Similarity'] > 60 and row['Title Similarity'] > 60) or (row['Content Similarity'] > 60 and row['Domain Similarity'] > 60) or (row['Content Similarity'] > 60 and row['Title Similarity'] > 60):
            status = 'phishing'
    elif 'Domain Similarity' in existing_columns and 'Title Similarity' in existing_columns:
        if row['Domain Similarity'] > 60 or row['Title Similarity'] > 60:
            status = 'suspected'
        if (row['Domain Similarity'] > 60 and row['Title Similarity'] > 60):
            status = 'phishing'
    elif 'Domain Similarity' in existing_columns and 'Content Similarity' in existing_columns:
        if row['Domain Similarity'] > 60 or row['Content Similarity'] > 60:
            status = 'suspected'
        if (row['Domain Similarity'] > 60 and row['Content Similarity'] > 60):
            status = 'phishing'
    elif 'Title Similarity' in existing_columns and 'Content Similarity' in existing_columns:
        if row['Title Similarity'] > 60 or row['Content Similarity'] > 60:
            status = 'suspected'
        if (row['Title Similarity'] > 60 and row['Content Similarity'] > 60):
            status = 'phishing'
    elif 'Domain Similarity' in existing_columns and row['Domain Similarity'] > 60:
        status = 'suspected'
    elif 'Title Similarity' in existing_columns and row['Title Similarity'] > 60:
        status = 'suspected'
    elif 'Content Similarity' in existing_columns and row['Content Similarity'] > 60:
        status = 'suspected'
    return status