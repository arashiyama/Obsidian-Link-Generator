#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_linker.py - Generate links between notes based on semantic similarity

This script reads all markdown notes in an Obsidian vault and:
1. Generates embeddings for each note
2. Calculates semantic similarity between notes
3. Adds links to semantically related notes in a "Related Notes" section

Features:
- Configurable similarity threshold
- Avoids duplicate links and preserves existing links

Author: Jonathan Care <jonc@lacunae.org>
"""

import os
import sys
import numpy as np
import re
import time
import json
import hashlib
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Set, TypedDict, cast
from dotenv import load_dotenv
from openai import OpenAI
import utils
import signal_handler
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Minimum similarity threshold for creating links
SIMILARITY_THRESHOLD = 0.75  # Adjusted for cosine similarity between OpenAI embeddings

# Batch size for embedding requests to avoid rate limits
EMBEDDING_BATCH_SIZE = 20

# Cache settings
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
EMBEDDINGS_CACHE_FILE = os.path.join(CACHE_DIR, "embeddings_cache.json")

# Ensure cache directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Type definitions for notes and related structures
class NoteMetadata(TypedDict, total=False):
    """Type for note metadata from frontmatter"""
    tags: List[str]
    category: str
    categories: List[str]
    type: str
    topic: str
    topics: List[str]
    project: str
    area: str

class Note(TypedDict):
    """Type for a note with content and metadata"""
    content: str
    content_for_embedding: str
    metadata: NoteMetadata
    content_with_metadata: Optional[str]

# Type for the dictionary of notes
NotesDict = Dict[str, Union[Note, str, Dict[str, Any]]]

# Type for embeddings cache
EmbeddingsCache = Dict[str, List[float]]

# Metadata settings
METADATA_WEIGHT = 0.3  # Weight to give metadata in similarity calculation (0-1)
IMPORTANT_METADATA_FIELDS = ["tags", "category", "categories", "type", "topic", "topics", "project", "area"]

def extract_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """
    Extract YAML frontmatter from note content.
    
    Args:
        content: String content of the note
        
    Returns:
        tuple: (frontmatter_dict, content_without_frontmatter)
    """
    # Check if content starts with YAML frontmatter (---)
    if content.startswith("---"):
        # Find the end of the frontmatter
        end_pos = content.find("---", 3)
        if end_pos != -1:
            frontmatter_text = content[3:end_pos].strip()
            content_without_frontmatter = content[end_pos+3:].strip()
            
            try:
                # Parse the YAML frontmatter
                frontmatter = yaml.safe_load(frontmatter_text)
                if not isinstance(frontmatter, dict):
                    frontmatter = {}
                
                return frontmatter, content_without_frontmatter
            except Exception as e:
                print(f"Error parsing frontmatter: {str(e)}")
    
    # No frontmatter or error parsing
    return {}, content

def load_notes(vault_path: Optional[str] = None) -> NotesDict:
    """
    Load all notes from the vault, extracting frontmatter metadata.
    
    Args:
        vault_path: Path to the Obsidian vault
        
    Returns:
        Dictionary mapping file paths to note data
    """
    if not vault_path:
        vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
        if not vault_path:
            print("Error: No vault path provided. Set OBSIDIAN_VAULT_PATH in .env")
            sys.exit(1)
    
    # Dictionary to store notes
    notes: NotesDict = {}
    
    # Walk through all directories and files in the vault
    for root, dirs, files in os.walk(vault_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.endswith(".md"):
                try:
                    path = os.path.join(root, file)
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    # Extract frontmatter metadata
                    frontmatter, content_without_frontmatter = extract_frontmatter(content)
                    
                    # Store content and metadata
                    notes[path] = cast(Note, {
                        "content": content,
                        "content_for_embedding": content_without_frontmatter,
                        "metadata": frontmatter,
                        "content_with_metadata": None  # Will be set later
                    })
                except Exception as e:
                    print(f"Error reading {file}: {str(e)}")
    
    print(f"Loaded {len(notes)} notes from vault")
    
    # Print metadata statistics
    notes_with_metadata = sum(1 for note in notes.values() 
                             if isinstance(note, dict) and note.get("metadata"))
    if len(notes) > 0:
        metadata_percentage = notes_with_metadata / len(notes) * 100
    else:
        metadata_percentage = 0.0
    print(f"Found {notes_with_metadata} notes with frontmatter metadata ({metadata_percentage:.1f}%)")
    
    return notes

@retry(
    retry=retry_if_exception_type((Exception)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    reraise=True
)
def get_embedding_batch(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for a batch of texts using OpenAI API with retry logic.
    
    Args:
        texts: List of text strings to get embeddings for
        
    Returns:
        List of embedding vectors (each vector is a list of floats)
    """
    try:
        # Truncate texts to stay within token limits
        truncated_texts = [text[:8000] for text in texts]
        
        response = client.embeddings.create(
            input=truncated_texts,
            model="text-embedding-3-small"  # Using the latest embedding model
        )
        
        # Extract embeddings from response
        embeddings = [item.embedding for item in response.data]
        return embeddings
    except Exception as e:
        print(f"Error getting embeddings batch: {str(e)}")
        raise

def get_content_hash(content: str) -> str:
    """
    Generate a hash for content to uniquely identify it for caching purposes.
    
    Args:
        content: String content to hash
        
    Returns:
        MD5 hash of the content as a hex string
    """
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def load_embeddings_cache() -> EmbeddingsCache:
    """
    Load the embeddings cache from disk.
    
    Returns:
        Dictionary mapping content hashes to embedding vectors
    """
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        try:
            with open(EMBEDDINGS_CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            
            # Convert cached embeddings back to proper format
            embeddings_cache: EmbeddingsCache = {}
            for content_hash, embedding in cache_data.items():
                embeddings_cache[content_hash] = embedding
                
            print(f"Loaded embedding cache with {len(embeddings_cache)} entries")
            return embeddings_cache
        except Exception as e:
            print(f"Error loading embedding cache: {str(e)}")
    
    print("No embedding cache found or error loading cache, starting with empty cache")
    return {}

def save_embeddings_cache(cache: EmbeddingsCache) -> None:
    """
    Save the embeddings cache to disk.
    
    Args:
        cache: Dictionary mapping content hashes to embedding vectors
    """
    try:
        with open(EMBEDDINGS_CACHE_FILE, 'w') as f:
            json.dump(cache, f)
        print(f"Saved embedding cache with {len(cache)} entries")
    except Exception as e:
        print(f"Error saving embedding cache: {str(e)}")

def prepare_metadata_for_embedding(metadata: Optional[Dict[str, Any]]) -> str:
    """
    Prepare metadata for embedding by extracting relevant fields and formatting them as text.
    
    Args:
        metadata: Dict containing frontmatter metadata
        
    Returns:
        str: Formatted metadata text
    """
    if not metadata:
        return ""
    
    metadata_parts: List[str] = []
    
    # Extract important metadata fields
    for field in IMPORTANT_METADATA_FIELDS:
        if field in metadata:
            value = metadata[field]
            
            # Handle different types of values
            if isinstance(value, list):
                # Join list values with commas
                metadata_parts.append(f"{field}: {', '.join(str(v) for v in value)}")
            elif isinstance(value, (str, int, float, bool)):
                metadata_parts.append(f"{field}: {value}")
    
    return "\n".join(metadata_parts)

def get_embeddings(notes: Union[NotesDict, List[str]]) -> Optional[np.ndarray]:
    """
    Generate embeddings for all notes using OpenAI API with caching.
    
    Args:
        notes: Dictionary of notes or list of note contents
        
    Returns:
        Numpy array of embeddings vectors or None if error occurs
    """
    try:
        # Load the cache
        embeddings_cache = load_embeddings_cache()
        
        # Track which notes need new embeddings
        new_contents: List[str] = []
        new_indices: List[int] = []
        content_hashes: List[str] = []
        all_embeddings: List[List[float]] = []
        
        # Handle both dictionary and list input formats
        if isinstance(notes, dict):
            note_items = [(path, notes[path]) for path in notes.keys()]
        else:
            # Handle when notes is a list (values from dictionary)
            note_items = [(i, note) for i, note in enumerate(notes)]
        
        # Check which notes are already in cache
        print("Checking embedding cache for notes...")
        for i, (path, note_data) in enumerate(note_items):
            
            # Prepare content for embedding with metadata enhancement
            # Handle both dictionary format from load_notes() and list format from obsidian_enhance.py
            if isinstance(note_data, dict) and "content_for_embedding" in note_data:
                content_for_embedding = note_data["content_for_embedding"]
                metadata = note_data.get("metadata", {})
            else:
                # When given a list of values from notes dictionary
                # Each item might be a string or a dict with "content" key
                if isinstance(note_data, dict) and "content" in note_data:
                    content = note_data["content"]
                else:
                    content = note_data if isinstance(note_data, str) else ""
                # Extract frontmatter to get metadata and content for embedding
                metadata, content_for_embedding = extract_frontmatter(content)
            
            metadata_text = prepare_metadata_for_embedding(metadata)
            
            # If we have metadata, add it to the content with appropriate weighting
            if metadata_text:
                # Repeat metadata text to give it appropriate weight in the embedding
                weighted_metadata = "\n\n" + metadata_text * 3  # Repeat to increase weight
                content_with_metadata = content_for_embedding + weighted_metadata
            else:
                content_with_metadata = content_for_embedding
                
            # Store the prepared content for embedding
            if isinstance(note_data, dict):
                note_data["content_with_metadata"] = content_with_metadata
            
            # Generate a hash for caching
            content_hash = get_content_hash(content_with_metadata)
            content_hashes.append(content_hash)
            
            if content_hash in embeddings_cache:
                # Use cached embedding
                all_embeddings.append(embeddings_cache[content_hash])
            else:
                # Mark for processing
                new_contents.append(content_with_metadata)
                new_indices.append(i)
        
        total_notes = len(note_items)
        cache_hits = total_notes - len(new_contents)
        if total_notes > 0:
            cache_hit_percentage = cache_hits / total_notes * 100
        else:
            cache_hit_percentage = 0.0
        print(f"Cache hits: {cache_hits}/{total_notes} notes ({cache_hit_percentage:.1f}%)")
        
        if new_contents:
            print(f"Generating embeddings for {len(new_contents)} new notes in batches of {EMBEDDING_BATCH_SIZE}")
            
            # Process new notes in batches
            new_embeddings: List[List[float]] = []
            for i in range(0, len(new_contents), EMBEDDING_BATCH_SIZE):
                batch = new_contents[i:i + EMBEDDING_BATCH_SIZE]
                print(f"Processing batch {i//EMBEDDING_BATCH_SIZE + 1}/{(len(new_contents) + EMBEDDING_BATCH_SIZE - 1)//EMBEDDING_BATCH_SIZE}")
                
                # Get embeddings for this batch
                batch_embeddings = get_embedding_batch(batch)
                new_embeddings.extend(batch_embeddings)
                
                # Brief pause to respect rate limits
                if i + EMBEDDING_BATCH_SIZE < len(new_contents):
                    time.sleep(1)
            
            # Update the cache with new embeddings
            for i, embedding in enumerate(new_embeddings):
                content_hash = content_hashes[new_indices[i]]
                embeddings_cache[content_hash] = embedding
            
            # Insert new embeddings at the correct positions
            for i, embedding in zip(new_indices, new_embeddings):
                all_embeddings.insert(i, embedding)
            
            # Save the updated cache
            save_embeddings_cache(embeddings_cache)
        
        # Convert to numpy array for easier manipulation
        try:
            # Make sure all embeddings have the same length for numpy conversion
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            return embeddings_array
        except ValueError as e:
            print(f"Error converting embeddings to numpy array: {str(e)}")
            print("This may be due to inconsistent embedding dimensions")
            # Try alternate approach by padding or truncating if needed
            max_len = max(len(emb) for emb in all_embeddings) if all_embeddings else 0
            normalized_embeddings: List[List[float]] = []
            for emb in all_embeddings:
                if len(emb) < max_len:
                    # Pad with zeros if shorter
                    padded = emb + [0.0] * (max_len - len(emb))
                    normalized_embeddings.append(padded)
                else:
                    # Use as is or truncate if needed
                    normalized_embeddings.append(emb[:max_len])
            
            return np.array(normalized_embeddings, dtype=np.float32)
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        return None

def cosine_similarity_matrix(embeddings: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Calculate cosine similarity between all pairs of embeddings.
    
    Args:
        embeddings: Numpy array of embedding vectors
        
    Returns:
        Similarity matrix as numpy array, or None if invalid input
    """
    # Check if embeddings is None or empty
    if embeddings is None or len(embeddings) == 0:
        print("Error: No valid embeddings to calculate similarity matrix")
        return None
    
    # Check for NaN or None values in the embeddings
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

def generate_links(notes: NotesDict, 
               embeddings: Optional[np.ndarray], 
               existing_links: Optional[Dict[str, List[str]]] = None, 
               subset_notes: Optional[Dict[str, bool]] = None) -> int:
    """
    Generate links based on semantic similarity from OpenAI embeddings.
    
    Args:
        notes: Dict mapping file paths to note data (content, metadata, etc.)
        embeddings: Numpy array of embeddings for all notes
        existing_links: Dict mapping file paths to lists of already-linked note names
        subset_notes: Dict of notes to update (if None, update all notes)
        
    Returns:
        Number of notes updated with new links
    """
    # Check for valid embeddings
    if embeddings is None or len(embeddings) == 0:
        print("Error: Cannot generate links - no valid embeddings provided")
        return 0
        
    updated = 0
    filenames = list(notes.keys())
    
    # If no existing links provided, create an empty dictionary
    if existing_links is None:
        existing_links = {}
        for path in notes:
            note_content = notes[path]["content"] if isinstance(notes[path], dict) else str(notes[path])
            existing_links[path] = utils.extract_existing_links(note_content)
    
    print("Calculating similarity matrix...")
    similarity_matrix = cosine_similarity_matrix(embeddings)
    if similarity_matrix is None:
        print("Error: Failed to calculate similarity matrix, cannot continue with link generation")
        return 0
    print("Similarity matrix calculated")
    
    # Process each note
    for idx, file in enumerate(filenames):
        # If subset_notes is provided, only process notes in the subset
        if subset_notes is not None and file not in subset_notes:
            continue
            
        try:
            # Get similarities from pre-calculated matrix
            similarities = similarity_matrix[idx]
            similarities[idx] = 0  # Set self-similarity to 0
            
            # Find notes with similarity above threshold
            related_indices = np.where(similarities > SIMILARITY_THRESHOLD)[0]
            
            # Get existing links for this note
            current_links = existing_links.get(file, [])
            
            # Create new link entries
            link_entries: List[str] = []
            
            # Extract existing section if it exists
            note_content = notes[file]["content"] if isinstance(notes[file], dict) else str(notes[file])
            section_text, _ = utils.extract_section(note_content, "## Related Notes")
            existing_link_entries: List[str] = []
            if section_text:
                existing_link_entries = section_text.split("\n")
            
            # Sort related notes by similarity score (descending)
            sorted_indices = sorted(related_indices, key=lambda i: similarities[i], reverse=True)
            
            # Limit to top 10 most similar notes
            top_indices = sorted_indices[:10]
            
            # Create entries for related notes
            for i in top_indices:
                # Get the note name without extension
                rel_path = filenames[i]
                note_name = utils.get_note_filename(rel_path)
                
                # Skip if already linked in the document
                if note_name in current_links:
                    continue
                
                # Add to current links to avoid duplicates in future iterations
                current_links.append(note_name)
                
                # Format the link with similarity score
                similarity = similarities[i]
                
                # Add metadata information if available
                metadata_info = ""
                shared_fields: List[str] = []
                
                # Extract metadata from related note
                rel_metadata: Dict[str, Any] = {}
                if isinstance(notes[rel_path], dict) and "metadata" in notes[rel_path]:
                    rel_metadata = notes[rel_path]["metadata"]
                else:
                    # Extract metadata if it's a string or dict with content
                    if isinstance(notes[rel_path], dict) and "content" in notes[rel_path]:
                        rel_content = notes[rel_path]["content"]
                    else:
                        rel_content = str(notes[rel_path])
                    rel_metadata, _ = extract_frontmatter(rel_content)
                
                # Extract metadata from current note
                curr_metadata: Dict[str, Any] = {}
                if isinstance(notes[file], dict) and "metadata" in notes[file]:
                    curr_metadata = notes[file]["metadata"]
                else:
                    # Extract metadata if it's a string or dict with content
                    if isinstance(notes[file], dict) and "content" in notes[file]:
                        curr_content = notes[file]["content"]
                    else:
                        curr_content = str(notes[file])
                    curr_metadata, _ = extract_frontmatter(curr_content)
                
                if rel_metadata and curr_metadata:
                    # Check for shared metadata fields
                    for field in IMPORTANT_METADATA_FIELDS:
                        if field in rel_metadata and field in curr_metadata:
                            rel_value = rel_metadata[field]
                            curr_value = curr_metadata[field]
                            
                            # Check for shared values
                            if isinstance(rel_value, list) and isinstance(curr_value, list):
                                # Find intersection of lists
                                shared = set([str(x).lower() for x in rel_value]) & set([str(x).lower() for x in curr_value])
                                if shared:
                                    shared_fields.append(f"{field}: {', '.join(shared)}")
                            elif str(rel_value).lower() == str(curr_value).lower():
                                shared_fields.append(f"{field}: {rel_value}")
                
                if shared_fields:
                    metadata_info = f" | Shared: {'; '.join(shared_fields)}"
                
                link_entry = f"- [[{note_name}]] (Semantic Similarity: {similarity:.2f}{metadata_info})"
                link_entries.append(link_entry)
            
            # If we have no entries to add and no existing entries, skip
            if not link_entries and not existing_link_entries:
                continue
            
            # Merge existing and new link entries
            all_link_entries = utils.merge_links(existing_link_entries, link_entries)
            
            # Update the section in the content
            note_content = notes[file]["content"] if isinstance(notes[file], dict) else str(notes[file])
            updated_content = utils.replace_section(
                note_content, 
                "## Related Notes", 
                "\n".join(all_link_entries)
            )
            
            # Save the updated content
            if isinstance(notes[file], dict):
                notes[file]["content"] = updated_content
            else:
                notes[file] = updated_content
            updated += 1
            
        except Exception as e:
            print(f"Error generating links for {file}: {str(e)}")
    
    print(f"Added semantic links to {updated} notes")
    return updated

def save_notes(notes: NotesDict, vault_path: Optional[str] = None) -> int:
    """
    Save updated notes to disk.
    
    Args:
        notes: Dictionary mapping file paths to note data
        vault_path: Optional path to the vault (not used but kept for API compatibility)
        
    Returns:
        Number of notes successfully saved
    """
    saved = 0
    errors = 0
    
    for path, note_data in notes.items():
        try:
            with open(path, "w", encoding="utf-8") as f:
                content = note_data["content"] if isinstance(note_data, dict) else str(note_data)
                f.write(content)
            saved += 1
        except Exception as e:
            print(f"Error saving {path}: {str(e)}")
            errors += 1
    
    print(f"Saved {saved} notes with {errors} errors")
    return saved

def cleanup_before_exit() -> None:
    """Clean up resources before exiting."""
    print("Performing cleanup before exit...")
    # Save any pending cache updates
    print("Semantic linking tool interrupted. No files have been modified.")
    print("Cleanup completed. Goodbye!")

def main() -> int:
    """
    Main function to run semantic linking.
    
    Returns:
        Number of notes processed
    """
    # Set up clean interrupt handling
    signal_handler.setup_interrupt_handling()
    
    # Register cleanup function
    signal_handler.register_cleanup_function(cleanup_before_exit)
    
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
    if not vault_path:
        print("Error: OBSIDIAN_VAULT_PATH not set in environment or .env file")
        sys.exit(1)
    
    print(f"Loading notes from vault: {vault_path}")
    notes = load_notes(vault_path)
    
    print("Generating embeddings for notes")
    embeddings = get_embeddings(notes)
    
    if embeddings is None:
        print("Failed to generate embeddings")
        sys.exit(1)
    
    print("Generating semantic links")
    generate_links(notes, embeddings)
    
    print("Saving notes")
    saved = save_notes(notes, vault_path)
    
    print(f"Added semantic links to {saved} notes")
    return saved

if __name__ == "__main__":
    main()
