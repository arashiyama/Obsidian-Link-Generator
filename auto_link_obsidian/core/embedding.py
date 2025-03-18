#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embedding.py - Abstract and concrete embedding providers for Auto Link Obsidian

This module defines the EmbeddingProvider interface and concrete implementations
for different embedding services (OpenAI, local, etc.).
"""

import os
import json
import hashlib
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union, Type
from pathlib import Path

from .config import config
from .note import Note


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    
    This interface allows swapping different embedding providers (OpenAI, local, etc.)
    without changing the rest of the code.
    """
    
    # Provider name - subclasses should override
    NAME = "base"
    
    def __init__(self):
        """Initialize the embedding provider."""
        # Load the embedding cache if it exists
        self.cache = self._load_cache()
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to get embeddings for
            
        Returns:
            Numpy array of embeddings
        """
        pass
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.
        
        Args:
            text: Text string to get embedding for
            
        Returns:
            Embedding vector as list of floats
        """
        pass
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        # Convert to numpy arrays
        a = np.array(embedding1)
        b = np.array(embedding2)
        
        # Calculate cosine similarity
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def embedding_exists(self, text: str) -> bool:
        """
        Check if an embedding for the given text exists in the cache.
        
        Args:
            text: Text to check for
            
        Returns:
            True if the embedding exists in cache, False otherwise
        """
        text_hash = self._get_hash(text)
        return text_hash in self.cache
    
    def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding from cache if it exists.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None if not found
        """
        text_hash = self._get_hash(text)
        if text_hash in self.cache:
            return self.cache[text_hash]
        return None
    
    def add_to_cache(self, text: str, embedding: List[float]) -> None:
        """
        Add an embedding to the cache.
        
        Args:
            text: Text that was embedded
            embedding: Embedding vector
        """
        text_hash = self._get_hash(text)
        self.cache[text_hash] = embedding
        self._save_cache()
    
    def _get_hash(self, text: str) -> str:
        """
        Generate a hash for a text string.
        
        Args:
            text: Text to hash
            
        Returns:
            MD5 hash of the text
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self) -> str:
        """
        Get path to the cache file.
        
        Returns:
            Path to the cache file
        """
        cache_file = f"{self.NAME}_embeddings_cache.json"
        return os.path.join(config["cache_dir_path"], cache_file)
    
    def _load_cache(self) -> Dict[str, List[float]]:
        """
        Load the embedding cache from disk.
        
        Returns:
            Dictionary mapping content hashes to embeddings
        """
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                
                print(f"Loaded {len(cache_data)} cached embeddings from {cache_path}")
                return cache_data
            except Exception as e:
                print(f"Error loading embedding cache: {str(e)}")
        
        print(f"No embedding cache found at {cache_path}, starting with empty cache")
        return {}
    
    def _save_cache(self) -> None:
        """Save the embedding cache to disk."""
        cache_path = self._get_cache_path()
        try:
            with open(cache_path, 'w') as f:
                json.dump(self.cache, f)
            print(f"Saved {len(self.cache)} embeddings to cache at {cache_path}")
        except Exception as e:
            print(f"Error saving embedding cache: {str(e)}")
    
    @classmethod
    def get_note_content_for_embedding(cls, note: Note) -> str:
        """
        Prepare note content for embedding by combining text and metadata.
        
        Args:
            note: Note object to prepare
            
        Returns:
            Prepared text for embedding
        """
        # Get frontmatter metadata as text
        metadata_text = cls.format_metadata_for_embedding(note.frontmatter)
        
        # If we have metadata, add it to the content with appropriate weighting
        if metadata_text:
            # Repeat metadata text to give it appropriate weight in the embedding
            weighted_metadata = "\n\n" + metadata_text * 3  # Repeat to increase weight
            return note.body + weighted_metadata
        else:
            return note.body
    
    @classmethod
    def format_metadata_for_embedding(cls, metadata: Dict[str, Any]) -> str:
        """
        Format metadata as text for embedding.
        
        Args:
            metadata: Dictionary containing metadata
            
        Returns:
            Formatted metadata text
        """
        if not metadata:
            return ""
        
        metadata_parts = []
        important_fields = config["important_metadata_fields"]
        
        # Extract important metadata fields
        for field in important_fields:
            if field in metadata:
                value = metadata[field]
                
                # Handle different types of values
                if isinstance(value, list):
                    # Join list values with commas
                    metadata_parts.append(f"{field}: {', '.join(str(v) for v in value)}")
                elif isinstance(value, (str, int, float, bool)):
                    metadata_parts.append(f"{field}: {value}")
        
        return "\n".join(metadata_parts)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using OpenAI API.
    """
    
    NAME = "openai"
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the OpenAI embedding provider.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model: OpenAI embedding model to use
        """
        super().__init__()
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Install it with 'pip install openai'.")
        
        self.model = model or config["embedding_model"] or "text-embedding-3-small"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment")
        
        self.client = OpenAI(api_key=self.api_key)
        self.batch_size = config["embedding_batch_size"] or 20
        
        print(f"Initialized OpenAI embedding provider with model {self.model}")
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts using OpenAI API.
        
        Args:
            texts: List of text strings to get embeddings for
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        # Check cache first
        all_embeddings = []
        new_texts = []
        new_indices = []
        
        # First pass: check cache
        for i, text in enumerate(texts):
            cached_embedding = self.get_cached_embedding(text)
            if cached_embedding is not None:
                all_embeddings.append(cached_embedding)
            else:
                new_texts.append(text)
                new_indices.append(i)
        
        cache_hits = len(texts) - len(new_texts)
        print(f"Cache hits: {cache_hits}/{len(texts)} ({cache_hits/len(texts)*100:.1f}%)")
        
        # Second pass: get new embeddings from API in batches
        if new_texts:
            print(f"Getting {len(new_texts)} new embeddings in batches of {self.batch_size}")
            
            new_embeddings = []
            for i in range(0, len(new_texts), self.batch_size):
                batch = new_texts[i:i + self.batch_size]
                print(f"Processing batch {i//self.batch_size + 1}/{(len(new_texts) + self.batch_size - 1)//self.batch_size}")
                
                # Truncate texts to stay within token limits
                truncated_batch = [text[:8000] for text in batch]
                
                try:
                    # Get embeddings from OpenAI API
                    response = self.client.embeddings.create(
                        input=truncated_batch,
                        model=self.model
                    )
                    
                    # Extract embeddings from response
                    batch_embeddings = [item.embedding for item in response.data]
                    new_embeddings.extend(batch_embeddings)
                    
                    # Add to cache
                    for text, embedding in zip(batch, batch_embeddings):
                        self.add_to_cache(text, embedding)
                    
                    # Small delay to respect rate limits
                    if i + self.batch_size < len(new_texts):
                        time.sleep(1)
                        
                except Exception as e:
                    print(f"Error getting embeddings from OpenAI API: {str(e)}")
                    # Try to continue with the embeddings we have
            
            # Insert new embeddings at the correct positions
            for i, embedding in zip(new_indices, new_embeddings):
                all_embeddings.insert(i, embedding)
        
        # Convert to numpy array
        try:
            return np.array(all_embeddings, dtype=np.float32)
        except ValueError as e:
            print(f"Error converting embeddings to numpy array: {str(e)}")
            # Try to handle inconsistent dimensions
            max_dim = max(len(emb) for emb in all_embeddings)
            normalized = []
            for emb in all_embeddings:
                if len(emb) < max_dim:
                    # Pad with zeros
                    normalized.append(emb + [0.0] * (max_dim - len(emb)))
                else:
                    normalized.append(emb[:max_dim])
            return np.array(normalized, dtype=np.float32)
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text using OpenAI API.
        
        Args:
            text: Text string to get embedding for
            
        Returns:
            Embedding vector as list of floats
        """
        # Check cache first
        cached_embedding = self.get_cached_embedding(text)
        if cached_embedding is not None:
            return cached_embedding
        
        # Truncate text to stay within token limits
        truncated_text = text[:8000]
        
        try:
            # Get embedding from OpenAI API
            response = self.client.embeddings.create(
                input=[truncated_text],
                model=self.model
            )
            
            # Extract embedding from response
            embedding = response.data[0].embedding
            
            # Add to cache
            self.add_to_cache(text, embedding)
            
            return embedding
        except Exception as e:
            print(f"Error getting embedding from OpenAI API: {str(e)}")
            # Return empty embedding as fallback
            return [0.0] * 1536  # Most OpenAI embedding models use 1536 dimensions


# Factory function to get the appropriate embedding provider
def get_embedding_provider(provider_type: str = "openai") -> EmbeddingProvider:
    """
    Get an embedding provider of the specified type.
    
    Args:
        provider_type: Type of embedding provider to use
        
    Returns:
        An instance of the requested embedding provider
    """
    providers = {
        "openai": OpenAIEmbeddingProvider,
    }
    
    if provider_type not in providers:
        print(f"Unknown embedding provider '{provider_type}', falling back to OpenAI")
        provider_type = "openai"
    
    return providers[provider_type]()