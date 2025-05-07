import os
import re

import numpy as np
import requests

from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings


response = requests.get(
    "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
)
response.raise_for_status()
faq_text = response.text

# This is another way to split text and generate documents

docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]

class VectorStoreRetriever:
    def __init__(self, text_docs: list, vectors:  list, embeddings_model: GoogleGenerativeAIEmbeddings):
        self._arr = np.array(vectors)
        self._docs = text_docs
        self._embeddings = embeddings_model

    @classmethod
    def from_docs(cls, text_docs: list[dict], embeddings_model: GoogleGenerativeAIEmbeddings):
        """ Get embeddings for all documents """
        vectors = embeddings_model.embed_documents([doc["page_content"] for doc in text_docs])
        return cls(text_docs, vectors, embeddings_model)

    def query(self, query: str, k: int = 5) -> list[dict]:
        """ Get embedding for the query """
        query_vector = np.array(self._embeddings.embed_query(query))

        # Calculate similarity score. @ from numpy do matrix multiplication
        scores = query_vector @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]

retriever = VectorStoreRetriever.from_docs(docs, GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"))

@tool
def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain options are permitted.
        Use this before making any flight changes performing other 'write' events."""
    text_docs = retriever.query(query, k=2)
    return "\n\n".join([doc["page_content"] for doc in text_docs])