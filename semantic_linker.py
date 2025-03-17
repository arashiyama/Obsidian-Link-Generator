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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import utils
import signal_handler

# Load environment variables from .env file
load_dotenv()

# Minimum similarity threshold for creating links
SIMILARITY_THRESHOLD = 0.3

def load_notes(vault_path=None):
    """Load all notes from the vault."""
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
                    
                    notes[path] = content
                except Exception as e:
                    print(f"Error reading {file}: {str(e)}")
    
    print(f"Loaded {len(notes)} notes from vault")
    return notes

def get_embeddings(contents):
    """Generate TF-IDF embeddings for all notes."""
    try:
        # Initialize the TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            analyzer='word',
            ngram_range=(1, 2),
            min_df=2
        )
        
        # Fit and transform the notes to get embeddings
        embeddings = vectorizer.fit_transform(contents)
        
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        return None

def generate_links(notes, embeddings, filenames, existing_links=None, subset_notes=None):
    """
    Generate links based on semantic similarity.
    
    Args:
        notes: Dict mapping file paths to note contents
        embeddings: Matrix of embeddings for all notes
        filenames: List of file paths corresponding to rows in embeddings
        existing_links: Dict mapping file paths to lists of already-linked note names
        subset_notes: Dict of notes to update (if None, update all notes)
    """
    updated = 0
    
    # If no existing links provided, create an empty dictionary
    if existing_links is None:
        existing_links = {}
        for path in notes:
            existing_links[path] = utils.extract_existing_links(notes[path])
    
    # Process each note
    for idx, file in enumerate(filenames):
        # If subset_notes is provided, only process notes in the subset
        if subset_notes is not None and file not in subset_notes:
            continue
            
        try:
            # Calculate similarities with all other notes
            similarities = cosine_similarity([embeddings[idx]], embeddings)[0]
            similarities[idx] = 0  # Set self-similarity to 0
            
            # Find notes with similarity above threshold
            related_indices = np.where(similarities > SIMILARITY_THRESHOLD)[0]
            
            # Get existing links for this note
            current_links = existing_links.get(file, [])
            
            # Create new link entries
            link_entries = []
            
            # Extract existing section if it exists
            section_text, _ = utils.extract_section(notes[file], "## Related Notes")
            existing_link_entries = []
            if section_text:
                existing_link_entries = section_text.split("\n")
            
            # Create entries for related notes
            for i in related_indices:
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
                link_entry = f"- [[{note_name}]] (Similarity: {similarity:.2f})"
                link_entries.append(link_entry)
            
            # If we have no entries to add and no existing entries, skip
            if not link_entries and not existing_link_entries:
                continue
            
            # Merge existing and new link entries
            all_link_entries = utils.merge_links(existing_link_entries, link_entries)
            
            # Update the section in the content
            updated_content = utils.replace_section(
                notes[file], 
                "## Related Notes", 
                "\n".join(all_link_entries)
            )
            
            # Save the updated content
            notes[file] = updated_content
            updated += 1
            
        except Exception as e:
            print(f"Error generating links for {file}: {str(e)}")
    
    print(f"Added semantic links to {updated} notes")
    return updated

def save_notes(notes, vault_path=None):
    """Save updated notes to disk."""
    saved = 0
    errors = 0
    
    for path, content in notes.items():
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            saved += 1
        except Exception as e:
            print(f"Error saving {path}: {str(e)}")
            errors += 1
    
    print(f"Saved {saved} notes with {errors} errors")
    return saved

def cleanup_before_exit():
    """Clean up resources before exiting."""
    print("Performing cleanup before exit...")
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
    filenames, contents = list(notes.keys()), list(notes.values())
    embeddings = get_embeddings(contents)
    
    if embeddings is None:
        print("Failed to generate embeddings")
        sys.exit(1)
    
    print("Generating semantic links")
    generate_links(notes, embeddings, filenames)
    
    print("Saving notes")
    saved = save_notes(notes, vault_path)
    
    print(f"Added semantic links to {saved} notes")
    return saved

if __name__ == "__main__":
    main()
