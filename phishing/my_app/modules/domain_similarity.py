import textdistance

# Function to strip TLD from domain, including multi-part TLDs like ".co.in"
def strip_tld(domain):
    multi_part_tlds = ['.co.in', '.org.in', '..in', '.in', '.com']
    for tld in multi_part_tlds:
        if domain.endswith(tld):
            return domain[:-len(tld)]
    parts = domain.split('.')
    if len(parts) > 2:
        return '.'.join(parts[:-1])
    return domain

# Function to remove specified substrings from the domain
def remove_substrings(domain, substrings):
    for substring in substrings:
        domain = domain.replace(substring, "")
    return domain

def calculate_domain_similarity(parent, child):
    if not parent or not child:
        return -1
    try:
        # Define substrings to be removed
        substrings_to_remove = [
            "xyz", "abc", "123", "online", "site", "shop", "store", "web", "info",
            "net", "my", "the", "best", "top", "pro", "plus", "gov", "free", "biz",
            "crt", "krt", 'india', 'mart', 'bank', 'customer', 'service', 'www.','credit'
        ]

        # Ensure parent and child are strings
        parent = str(parent)
        child = str(child)

        # Remove specified substrings
        parent_cleaned = remove_substrings(parent, substrings_to_remove)
        child_cleaned = remove_substrings(child, substrings_to_remove)

        # Calculate Damerau-Levenshtein similarity with TLD
        levenshtein_similarity_with_TLD = textdistance.damerau_levenshtein.normalized_similarity(parent_cleaned, child_cleaned) * 100

        # Calculate Damerau-Levenshtein similarity without TLD
        parent_stripped = strip_tld(parent_cleaned)
        child_stripped = strip_tld(child_cleaned)
        levenshtein_similarity_without_TLD = textdistance.damerau_levenshtein.normalized_similarity(parent_stripped, child_stripped) * 100

        # Calculate additional similarity metrics only if the stripped Levenshtein similarity is below a threshold
        jaccard_similarity_with_TLD = 0
        # Calculate positional Jaccard similarity with TLD
        parent_set = set(parent_cleaned)
        child_set = set(child_cleaned)
        intersection_count = len(parent_set.intersection(child_set))
        union_count = len(parent_set.union(child_set))
        jaccard_similarity_with_TLD = (intersection_count / union_count) * 100

        jaccard_similarity_without_TLD = 0

        # Calculate positional Jaccard similarity without TLD
        parent_without_tld_set = set(parent_stripped)
        child_without_tld_set = set(child_stripped)
        intersection_count = len(parent_without_tld_set.intersection(child_without_tld_set))
        union_count = len(parent_without_tld_set.union(child_without_tld_set))
        jaccard_similarity_without_TLD = (intersection_count / union_count) * 100

        # Use a weighted average to combine the similarities
        combined_similarity = (0.35 * levenshtein_similarity_without_TLD + 0.30 * jaccard_similarity_without_TLD + 0.05 * levenshtein_similarity_with_TLD + 0.30 * jaccard_similarity_with_TLD)

        return min(combined_similarity, 100)  # Ensure similarity does not exceed 100
    except Exception as e:
        print(f"Error calculating domain similarity: {e}")
        return -1