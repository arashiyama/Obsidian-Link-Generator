#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_linker.py - Generates semantic links between Obsidian notes using OpenAI embeddings

This script analyzes all notes in an Obsidian vault, computes embeddings for each note,
and creates "Related Notes" sections based on semantic similarity between notes.
Features include chunking long content, caching embeddings, and parallel processing.

Author: Jonathan Care <jonc@lacunae.org>
"""

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from tqdm import tqdm
import re
from openai import OpenAI
import tiktoken
import time
import json
import hashlib
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import backoff

load_dotenv()

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "/Users/jonc/Obsidian/Jonathans Brain")
EMBEDDING_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.8  # Adjust to refine the strength of connections
MAX_TOKENS = 8000  # Leave some buffer below the 8192 limit
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
MAX_WORKERS = min(32, multiprocessing.cpu_count() * 2)  # Use at most 2 threads per CPU core, max 32

# Create cache directory if it doesn't exist
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")  # This encoding works for text-embedding-3-small

def count_tokens(text):
    """Count the number of tokens in a text string."""
    return len(tokenizer.encode(text))

def chunk_text(text, max_tokens=MAX_TOKENS):
    """Split text into chunks of max_tokens."""
    # Simple approach: split by paragraphs first
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed the limit, save current chunk and start a new one
        if count_tokens(current_chunk + paragraph) > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    # If any single paragraph is too long, we need to split it further
    result = []
    for chunk in chunks:
        if count_tokens(chunk) > max_tokens:
            # Split by sentences if a paragraph is too long
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            current_chunk = ""
            
            for sentence in sentences:
                if count_tokens(current_chunk + sentence) > max_tokens and current_chunk:
                    result.append(current_chunk)
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
            
            if current_chunk:
                if count_tokens(current_chunk) > max_tokens:
                    # If a single sentence is still too long, split by character count
                    words = current_chunk.split()
                    current_chunk = ""
                    
                    for word in words:
                        test_chunk = current_chunk + " " + word if current_chunk else word
                        if count_tokens(test_chunk) > max_tokens:
                            result.append(current_chunk)
                            current_chunk = word
                        else:
                            current_chunk = test_chunk
                    
                    if current_chunk:
                        # Final check - if a single word is too long, split it by characters
                        if count_tokens(current_chunk) > max_tokens:
                            chars = list(current_chunk)
                            current_chunk = ""
                            
                            for char in chars:
                                test_chunk = current_chunk + char
                                if count_tokens(test_chunk) > max_tokens:
                                    result.append(current_chunk)
                                    current_chunk = char
                                else:
                                    current_chunk = test_chunk
                            
                            if current_chunk:
                                result.append(current_chunk)
                        else:
                            result.append(current_chunk)
                else:
                    result.append(current_chunk)
        else:
            result.append(chunk)
    
    # Final check to ensure no chunk exceeds the token limit
    final_result = []
    for chunk in result:
        if count_tokens(chunk) <= max_tokens:
            final_result.append(chunk)
        else:
            print(f"Warning: Chunk still exceeds token limit with {count_tokens(chunk)} tokens. Skipping.")
    
    return final_result

def load_notes(vault_path):
    notes = {}
    for root, dirs, files in os.walk(vault_path):
        # Modify dirs in-place to exclude venv directories
        dirs[:] = [d for d in dirs if d != "venv"]
        
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        notes[file] = f.read()
                except Exception as e:
                    print(f"Error reading file {path}: {str(e)}")
    return notes

def generate_cache_key(text):
    """Generate a unique cache key based on the content."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_embedding_from_cache(cache_key):
    """Retrieve embedding from cache if it exists."""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading cache file: {str(e)}")
    return None

def save_embedding_to_cache(cache_key, embedding):
    """Save embedding to cache."""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        with open(cache_file, 'w') as f:
            json.dump(embedding, f)
    except Exception as e:
        print(f"Error writing to cache file: {str(e)}")

@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    max_time=60,
    giveup=lambda e: "invalid_request_error" in str(e),  # Don't retry invalid requests
)
def get_embedding_with_backoff(chunk):
    """Get embedding with exponential backoff for rate limits."""
    response = client.embeddings.create(
        input=chunk,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def process_chunk(chunk):
    """Process a single chunk to get its embedding, with caching."""
    cache_key = generate_cache_key(chunk)
    cached_embedding = get_embedding_from_cache(cache_key)
    
    if cached_embedding:
        return cached_embedding
    
    try:
        embedding = get_embedding_with_backoff(chunk)
        save_embedding_to_cache(cache_key, embedding)
        return embedding
    except Exception as e:
        print(f"Error embedding chunk (length: {count_tokens(chunk)} tokens): {str(e)}")
        # Return a zero vector if embedding fails
        return [0.0] * 1536

def get_embeddings(texts):
    all_embeddings = []
    
    for text in tqdm(texts, desc="Generating embeddings"):
        # Check if the text as a whole is cached
        cache_key = generate_cache_key(text)
        cached_embedding = get_embedding_from_cache(cache_key)
        
        if cached_embedding:
            all_embeddings.append(cached_embedding)
            continue
            
        # Check if the text might be too long
        if count_tokens(text) > MAX_TOKENS:
            # Split into chunks
            chunks = chunk_text(text)
            
            # Process chunks in parallel
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                chunk_embeddings = list(executor.map(process_chunk, chunks))
            
            # Average the chunk embeddings
            if chunk_embeddings:
                avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
                all_embeddings.append(avg_embedding)
                # Cache the averaged embedding
                save_embedding_to_cache(cache_key, avg_embedding)
            else:
                # Fallback if all chunks failed
                all_embeddings.append([0.0] * 1536)
        else:
            # Text is short enough to embed directly
            embedding = process_chunk(text)
            all_embeddings.append(embedding)
    
    return np.array(all_embeddings)

def generate_links(notes, embeddings, filenames, existing_links=None):
    """
    Generate semantic links between notes based on embedding similarity.
    
    Args:
        notes: Dictionary of notes (path -> content)
        embeddings: Array of note embeddings
        filenames: List of filenames corresponding to embeddings
        existing_links: Dictionary of existing links for each note (path -> [links])
    """
    similarities = cosine_similarity(embeddings)
    np.fill_diagonal(similarities, 0)
    
    # If no existing links provided, create an empty dictionary
    if existing_links is None:
        existing_links = {}
        # Extract links from all notes
        for file, content in notes.items():
            existing_links[file] = extract_existing_links(content)

    for idx, file in enumerate(filenames):
        # Skip if file is not in notes (shouldn't happen but just in case)
        if file not in notes:
            continue
            
        # Get existing links for this note
        current_links = existing_links.get(file, [])
        
        # Find related notes
        related_indices = np.where(similarities[idx] > SIMILARITY_THRESHOLD)[0]
        
        # Get note names from indices, avoiding duplicates with existing links
        new_links = []
        for i in related_indices:
            # Extract note name without extension
            note_name = os.path.splitext(os.path.basename(filenames[i]))[0]
            
            # Skip if this note is already linked anywhere in the document
            if note_name in current_links:
                continue
                
            new_links.append(f"[[{note_name}]]")
            
            # Add to current links to avoid duplicates in future iterations
            current_links.append(note_name)
        
        # Extract existing related notes section if it exists
        existing_link_entries = []
        if "## Related Notes" in notes[file]:
            # Extract existing related notes section
            existing_section = re.search(r"## Related Notes\n(.*?)(?=\n## |\n#|\Z)", 
                                       notes[file], flags=re.DOTALL)
            if existing_section:
                # Extract existing links
                for line in existing_section.group(1).split("\n"):
                    if line.strip():
                        link_match = re.search(r'- \[\[(.*?)\]\]', line)
                        if link_match:
                            existing_link_entries.append(line)
        
        # Combine all links to be included in the section
        all_links = []
        
        # Add existing links first
        note_names_added = set()
        for entry in existing_link_entries:
            link_match = re.search(r'- \[\[(.*?)\]\]', entry)
            if link_match:
                note_name = link_match.group(1)
                if note_name not in note_names_added:
                    all_links.append(entry)
                    note_names_added.add(note_name)
        
        # Add new links
        for link in new_links:
            note_name = link[2:-2]  # Extract note name without brackets
            if note_name not in note_names_added:
                all_links.append(f"- {link}")
                note_names_added.add(note_name)
        
        # Skip if we have no links to add
        if not all_links:
            continue
            
        # Create the updated section
        link_section = "\n\n## Related Notes\n" + "\n".join(all_links)
        
        # Update note content
        if "## Related Notes" in notes[file]:
            # Replace existing section
            notes[file] = re.sub(r"## Related Notes.*?(?=\n## |\n#|\Z)", 
                               link_section, notes[file], flags=re.DOTALL)
        else:
            # Add new section
            notes[file] += link_section

def extract_existing_links(content):
    """Extract all existing wiki links from a note."""
    links = []
    
    # Find all wiki links in the content
    for match in re.finditer(r'\[\[(.*?)(?:\|.*?)?\]\]', content):
        link = match.group(1).strip()
        links.append(link)
    
    return links

def save_notes(notes, vault_path):
    for file, content in notes.items():
        path = os.path.join(vault_path, file)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            print(f"Error writing to file {path}: {str(e)}")

if __name__ == "__main__":
    # Check if cache directory exists
    print(f"Cache directory: {CACHE_DIR}")
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(f"Created cache directory at {CACHE_DIR}")
    
    print(f"Loading notes from {VAULT_PATH}...")
    notes = load_notes(VAULT_PATH)
    print(f"Loaded {len(notes)} notes")
    
    filenames, contents = list(notes.keys()), list(notes.values())
    print("Generating embeddings...")
    embeddings = get_embeddings(contents)
    
    print("Finding similar notes...")
    generate_links(notes, embeddings, filenames)
    
    print("Saving updated notes...")
    save_notes(notes, VAULT_PATH)
    
    print("✅ Semantic linking completed!")
