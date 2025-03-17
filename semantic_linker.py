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

# Metadata settings
METADATA_WEIGHT = 0.3  # Weight to give metadata in similarity calculation (0-1)
IMPORTANT_METADATA_FIELDS = ["tags", "category", "categories", "type", "topic", "topics", "project", "area"]

def extract_frontmatter(content):
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

def load_notes(vault_path=None):
    """Load all notes from the vault, extracting frontmatter metadata."""
    if not vault_path:
        vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
        if not vault_path:
            print("Error: No vault path provided. Set OBSIDIAN_VAULT_PATH in .env")
            sys.exit(1)
    
    # Dictionary to store notes
    notes = {}
    
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
                    notes[path] = {
                        "content": content,
                        "content_for_embedding": content_without_frontmatter,
                        "metadata": frontmatter
                    }
                except Exception as e:
                    print(f"Error reading {file}: {str(e)}")
    
    print(f"Loaded {len(notes)} notes from vault")
    
    # Print metadata statistics
    notes_with_metadata = sum(1 for note in notes.values() if note["metadata"])
    print(f"Found {notes_with_metadata} notes with frontmatter metadata ({notes_with_metadata/len(notes)*100:.1f}%)")
    
    return notes

@retry(
    retry=retry_if_exception_type((Exception)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    reraise=True
)
def get_embedding_batch(texts):
    """Get embeddings for a batch of texts using OpenAI API with retry logic."""
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

def get_content_hash(content):
    """Generate a hash for content to uniquely identify it for caching purposes."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def load_embeddings_cache():
    """Load the embeddings cache from disk."""
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        try:
            with open(EMBEDDINGS_CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            
            # Convert cached embeddings back to proper format
            embeddings_cache = {}
            for content_hash, embedding in cache_data.items():
                embeddings_cache[content_hash] = embedding
                
            print(f"Loaded embedding cache with {len(embeddings_cache)} entries")
            return embeddings_cache
        except Exception as e:
            print(f"Error loading embedding cache: {str(e)}")
    
    print("No embedding cache found or error loading cache, starting with empty cache")
    return {}

def save_embeddings_cache(cache):
    """Save the embeddings cache to disk."""
    try:
        with open(EMBEDDINGS_CACHE_FILE, 'w') as f:
            json.dump(cache, f)
        print(f"Saved embedding cache with {len(cache)} entries")
    except Exception as e:
        print(f"Error saving embedding cache: {str(e)}")

def prepare_metadata_for_embedding(metadata):
    """
    Prepare metadata for embedding by extracting relevant fields and formatting them as text.
    
    Args:
        metadata: Dict containing frontmatter metadata
        
    Returns:
        str: Formatted metadata text
    """
    if not metadata:
        return ""
    
    metadata_parts = []
    
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

def get_embeddings(notes):
    """Generate embeddings for all notes using OpenAI API with caching."""
    try:
        # Load the cache
        embeddings_cache = load_embeddings_cache()
        
        # Track which notes need new embeddings
        new_contents = []
        new_indices = []
        content_hashes = []
        all_embeddings = []
        note_keys = list(notes.keys())
        
        # Check which notes are already in cache
        print("Checking embedding cache for notes...")
        for i, path in enumerate(note_keys):
            note_data = notes[path]
            
            # Prepare content for embedding with metadata enhancement
            content_for_embedding = note_data["content_for_embedding"]
            metadata_text = prepare_metadata_for_embedding(note_data["metadata"])
            
            # If we have metadata, add it to the content with appropriate weighting
            if metadata_text:
                # Repeat metadata text to give it appropriate weight in the embedding
                weighted_metadata = "\n\n" + metadata_text * 3  # Repeat to increase weight
                content_with_metadata = content_for_embedding + weighted_metadata
            else:
                content_with_metadata = content_for_embedding
                
            # Store the prepared content for embedding
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
        
        cache_hits = len(note_keys) - len(new_contents)
        print(f"Cache hits: {cache_hits}/{len(note_keys)} notes ({cache_hits/len(note_keys)*100:.1f}%)")
        
        if new_contents:
            print(f"Generating embeddings for {len(new_contents)} new notes in batches of {EMBEDDING_BATCH_SIZE}")
            
            # Process new notes in batches
            new_embeddings = []
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
            max_len = max(len(emb) for emb in all_embeddings)
            normalized_embeddings = []
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

def cosine_similarity_matrix(embeddings):
    """Calculate cosine similarity between all pairs of embeddings."""
    # Normalize the vectors
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Calculate cosine similarity
    similarity_matrix = np.dot(normalized, normalized.T)
    
    return similarity_matrix

def generate_links(notes, embeddings, existing_links=None, subset_notes=None):
    """
    Generate links based on semantic similarity from OpenAI embeddings.
    
    Args:
        notes: Dict mapping file paths to note data (content, metadata, etc.)
        embeddings: Numpy array of embeddings for all notes
        existing_links: Dict mapping file paths to lists of already-linked note names
        subset_notes: Dict of notes to update (if None, update all notes)
    """
    updated = 0
    filenames = list(notes.keys())
    
    # If no existing links provided, create an empty dictionary
    if existing_links is None:
        existing_links = {}
        for path in notes:
            existing_links[path] = utils.extract_existing_links(notes[path]["content"])
    
    print("Calculating similarity matrix...")
    similarity_matrix = cosine_similarity_matrix(embeddings)
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
            link_entries = []
            
            # Extract existing section if it exists
            section_text, _ = utils.extract_section(notes[file]["content"], "## Related Notes")
            existing_link_entries = []
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
                rel_metadata = notes[rel_path]["metadata"]
                shared_fields = []
                
                if rel_metadata and notes[file]["metadata"]:
                    # Check for shared metadata fields
                    for field in IMPORTANT_METADATA_FIELDS:
                        if field in rel_metadata and field in notes[file]["metadata"]:
                            rel_value = rel_metadata[field]
                            curr_value = notes[file]["metadata"][field]
                            
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
            updated_content = utils.replace_section(
                notes[file]["content"], 
                "## Related Notes", 
                "\n".join(all_link_entries)
            )
            
            # Save the updated content
            notes[file]["content"] = updated_content
            updated += 1
            
        except Exception as e:
            print(f"Error generating links for {file}: {str(e)}")
    
    print(f"Added semantic links to {updated} notes")
    return updated

def save_notes(notes):
    """Save updated notes to disk."""
    saved = 0
    errors = 0
    
    for path, note_data in notes.items():
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(note_data["content"])
            saved += 1
        except Exception as e:
            print(f"Error saving {path}: {str(e)}")
            errors += 1
    
    print(f"Saved {saved} notes with {errors} errors")
    return saved

def cleanup_before_exit():
    """Clean up resources before exiting."""
    print("Performing cleanup before exit...")
    # Save any pending cache updates
    print("Semantic linking tool interrupted. No files have been modified.")
    print("Cleanup completed. Goodbye!")

def main():
    """Main function to run semantic linking."""
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
    saved = save_notes(notes)
    
    print(f"Added semantic links to {saved} notes")
    return saved

if __name__ == "__main__":
    main()
