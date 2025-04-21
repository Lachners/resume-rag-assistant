from keybert import KeyBERT

# Keyword extraction function for resume text only
def extract_keywords_only(text, top_n=20):
    kw_model = KeyBERT("all-MiniLM-L6-v2")
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return [kw[0].lower() for kw in keywords]
