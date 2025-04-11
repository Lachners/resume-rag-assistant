import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VectorSpace:
    def __init__(self, documents):
        """
        Initialize the vector space with a list of documents.
        :param documents: List of strings (documents)
        """
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = self.vectorizer.fit_transform(documents)
        self.documents = documents

    def embed_query(self, query):
        """
        Embed a query into the same vector space as the documents.
        :param query: String (query)
        :return: Sparse vector representation of the query
        """
        return self.vectorizer.transform([query])

    def search(self, query, top_k=5):
        """
        Perform similarity search for the query against the documents.
        :param query: String (query)
        :param top_k: Number of top results to return
        :return: List of tuples (document_index, similarity_score)
        """
        query_vector = self.embed_query(query)
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(index, similarities[index]) for index in top_indices]

# Example usage
if __name__ == "__main__":
    documents = [
        "Machine learning is fascinating.",
        "Artificial intelligence and machine learning are closely related.",
        "Natural language processing is a subfield of AI.",
        "I love programming in Python.",
        "Data science involves statistics and programming."
    ]

    vector_space = VectorSpace(documents)
    query = "Tell me about machine learning"
    results = vector_space.search(query, top_k=3)

    print("Query:", query)
    print("Top results:")
    for idx, score in results:
        print(f"Document: {documents[idx]} | Similarity: {score:.4f}")