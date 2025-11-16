import requests
import json
import numpy as np
from typing import List

class OllamaEmbeddingClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "embeddinggemma:latest"):
        self.base_url = base_url
        self.model = model
        self.embedding_url = f"{base_url}/api/embeddings"
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text string"""
        try:
            payload = {
                "model": self.model,
                "prompt": text
            }
            
            response = requests.post(self.embedding_url, json=payload,)
            response.raise_for_status()
            
            result = response.json()
            return result.get('embedding', [])
            
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return a zero vector as fallback (adjust dimension as needed)
            return [0.0] * 1024  # embeddinggemma typically uses 1024 dimensions
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (sequential processing)"""
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings

