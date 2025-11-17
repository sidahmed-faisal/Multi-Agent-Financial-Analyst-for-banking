import os
import requests
from typing import List


class OllamaEmbeddingClient:
    """Client for Ollama embeddings that reads defaults from environment.

    Environment variables:
    - OLLAMA_URL (default: http://localhost:11434)
    - OLLAMA_MODEL (default: nomic-embed-text:latest)
    """
    def __init__(self, base_url: str = None, model: str = None):
        # Allow explicit override via constructor, otherwise read from env, then fallback to hard-coded default
        self.base_url = base_url or os.environ.get('OLLAMA_URL', 'http://localhost:11434')
        self.model = model or os.environ.get('OLLAMA_MODEL', 'nomic-embed-text:latest')
        self.embedding_url = f"{self.base_url.rstrip('/')}/api/embeddings"

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text string"""
        try:
            payload = {
                "model": self.model,
                "prompt": text
            }

            response = requests.post(self.embedding_url, json=payload)
            response.raise_for_status()

            result = response.json()
            return result.get('embedding', [])

        except Exception as e:
            print(f"Error getting embedding from {self.embedding_url}: {e}")
            # Return a zero vector as fallback (adjust dimension as needed)
            return [0.0] * 768  # nomic-embed-text uses 768 dimensions

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (sequential processing)"""
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings

