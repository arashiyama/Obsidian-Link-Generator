#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic.py - Semantic linking implementation for Auto Link Obsidian

This module implements semantic linking between notes based on embedding similarity.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Set, Any

from ..core.note import Note
from ..core.config import config
from ..core.embedding import get_embedding_provider, EmbeddingProvider
from ..utils.markdown import extract_links
from .base_linker import BaseLinker, linker_registry


class SemanticLinker(BaseLinker):
    """
    Linker implementation that connects notes based on semantic similarity.
    """
    
    TYPE = "semantic"
    SECTION_HEADER = "## Related Notes"
    
    def __init__(self, vault_path: Optional[str] = None):
        """
        Initialize the semantic linker.
        
        Args:
            vault_path: Path to the Obsidian vault
        """
        super().__init__(vault_path)
        
        # Get the embedding provider
        self.embedding_provider = get_embedding_provider("openai")
        
        # Get the similarity threshold from config
        self.similarity_threshold = config["similarity_threshold"]
        
        # Print configuration info
        print(f"Initialized semantic linker with similarity threshold {self.similarity_threshold}")
    
    def process_notes(self) -> int:
        """
        Process notes to generate links based on semantic similarity.
        
        Returns:
            Number of notes processed
        """
        # Filter notes to determine which ones need processing
        notes_to_process = self.filter_notes_to_process()
        
        if not notes_to_process:
            print("No notes to process")
            return 0
            
        print(f"Processing {len(notes_to_process)} notes out of {len(self.notes)} total")
        
        # Prepare all notes for embedding
        print("Preparing notes for embedding...")
        texts = []
        for note in self.notes.values():
            text = EmbeddingProvider.get_note_content_for_embedding(note)
            texts.append(text)
        
        # Generate embeddings for all notes
        print("Generating embeddings for all notes (required for accurate similarity calculation)...")
        embeddings = self.embedding_provider.get_embeddings(texts)
        
        if embeddings is None or len(embeddings) == 0:
            print("Failed to generate embeddings")
            return 0
        
        # Calculate similarity matrix
        print("Calculating similarity matrix...")
        similarity_matrix = self._calculate_similarity_matrix(embeddings)
        
        # Generate links for notes that need processing
        processed_count = 0
        for note_path, note in notes_to_process.items():
            try:
                # Find the index of this note in the full notes dictionary
                note_idx = list(self.notes.keys()).index(note_path)
                
                # Get similarities from the precomputed matrix
                similarities = similarity_matrix[note_idx]
                
                # Exclude self-similarity
                similarities[note_idx] = 0
                
                # Find notes with similarity above threshold
                related_indices = np.where(similarities > self.similarity_threshold)[0]
                
                # Sort by similarity score (descending)
                sorted_indices = sorted(related_indices, key=lambda i: similarities[i], reverse=True)
                
                # Limit to top 10 most similar notes
                top_indices = sorted_indices[:10]
                
                if not top_indices:
                    continue
                
                # Get existing links in the note
                existing_links = extract_links(note.content)
                
                # Format link entries for related notes
                link_entries = []
                for i in top_indices:
                    # Get the related note
                    related_path = list(self.notes.keys())[i]
                    related_note = self.notes[related_path]
                    
                    # Skip if already linked
                    if related_note.title in existing_links:
                        continue
                    
                    # Format the link with similarity score
                    similarity = similarities[i]
                    
                    # Add metadata information if available
                    metadata_info = self._get_shared_metadata_info(note, related_note)
                    
                    link_entry = f"- [[{related_note.title}]] (Semantic Similarity: {similarity:.2f}{metadata_info})"
                    link_entries.append(link_entry)
                
                # Only update if we have links to add
                if link_entries:
                    # Add links to the note
                    note.add_links(link_entries, self.SECTION_HEADER)
                    processed_count += 1
                
            except Exception as e:
                print(f"Error processing note {note_path}: {str(e)}")
        
        print(f"Added semantic links to {processed_count} notes")
        return processed_count
    
    def _calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between all pairs of embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            Similarity matrix
        """
        # Check for NaN or None values
        if np.isnan(embeddings).any() or None in embeddings.flatten():
            print("Warning: NaN or None values found in embeddings, replacing with zeros")
            embeddings = np.nan_to_num(embeddings, nan=0.0)
        
        # Verify we have valid embeddings with non-zero norms
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            print("Warning: Zero-length embedding vectors found, adding small epsilon to avoid division by zero")
            epsilon = 1e-8
            norms = norms + epsilon
        
        # Normalize the vectors
        normalized = embeddings / norms
        
        # Calculate cosine similarity
        similarity_matrix = np.dot(normalized, normalized.T)
        
        return similarity_matrix
    
    def _get_shared_metadata_info(self, note1: Note, note2: Note) -> str:
        """
        Get formatted string with shared metadata between two notes.
        
        Args:
            note1: First note
            note2: Second note
            
        Returns:
            Formatted string with shared metadata
        """
        metadata_info = ""
        shared_fields = []
        
        for field in config["important_metadata_fields"]:
            if field in note1.frontmatter and field in note2.frontmatter:
                value1 = note1.frontmatter[field]
                value2 = note2.frontmatter[field]
                
                # Check for shared values
                if isinstance(value1, list) and isinstance(value2, list):
                    # Find intersection of lists
                    shared = set([str(x).lower() for x in value1]) & set([str(x).lower() for x in value2])
                    if shared:
                        shared_fields.append(f"{field}: {', '.join(shared)}")
                elif str(value1).lower() == str(value2).lower():
                    shared_fields.append(f"{field}: {value1}")
        
        if shared_fields:
            metadata_info = f" | Shared: {'; '.join(shared_fields)}"
        
        return metadata_info


# Register the linker
linker_registry["semantic"] = SemanticLinker