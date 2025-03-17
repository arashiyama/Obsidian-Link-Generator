#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
note_categorizer.py - Organize notes into categories using GenAI

This script analyzes note content and suggests categories for organizing
notes within an Obsidian vault.

Features:
- Discovers notes within a vault
- Uses OpenAI to suggest categories based on content
- Updates notes with category metadata
- Maintains a taxonomy of categories

Author: Jonathan Care <jonc@lacunae.org>
"""

import os
import sys
import re
import json
import random
import argparse
from dotenv import load_dotenv
from openai import OpenAI
import utils
import signal_handler

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
                    
                    notes[path] = {
                        "filename": file,
                        "content": content
                    }
                except Exception as e:
                    print(f"Error reading {file}: {str(e)}")
    
    print(f"Loaded {len(notes)} notes from vault")
    return notes

def save_notes(notes, vault_path=None):
    """Save all modified notes back to disk.
    
    This function is needed by obsidian_enhance.py to save notes after processing.
    """
    saved_count = 0
    error_count = 0
    
    try:
        for path, note_data in notes.items():
            try:
                # Get the content to save
                content = note_data["content"] if isinstance(note_data, dict) else note_data
                
                # Save the note
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                saved_count += 1
            except Exception as e:
                print(f"Error saving note {path}: {str(e)}")
                error_count += 1
                
        print(f"Saved {saved_count} notes, encountered {error_count} errors")
        return saved_count
    except Exception as e:
        print(f"Error in save_notes: {str(e)}")
        return 0

def load_or_create_category_taxonomy(taxonomy_path):
    """Load existing category taxonomy or create a new one."""
    if os.path.exists(taxonomy_path):
        try:
            with open(taxonomy_path, "r", encoding="utf-8") as f:
                taxonomy = json.load(f)
            print(f"Loaded existing taxonomy with {len(taxonomy)} categories")
            return taxonomy
        except Exception as e:
            print(f"Error loading taxonomy: {str(e)}")
    
    # Create new taxonomy
    print("Creating new category taxonomy")
    return {}

def save_category_taxonomy(taxonomy, taxonomy_path):
    """Save the category taxonomy to a file."""
    try:
        with open(taxonomy_path, "w", encoding="utf-8") as f:
            json.dump(taxonomy, f, indent=2)
        print(f"Saved taxonomy with {len(taxonomy)} categories")
    except Exception as e:
        print(f"Error saving taxonomy: {str(e)}")

def categorize_note(note_content, note_title, existing_taxonomy):
    """Use OpenAI to suggest categories for a note."""
    try:
        # Extract existing categories if any
        match = re.search(r'categories:\s*\[(.*?)\]', note_content)
        existing_categories = []
        if match:
            # Parse categories from the metadata
            category_string = match.group(1)
            existing_categories = [c.strip(' "\'') for c in category_string.split(',')]
        
        # Create a string representation of existing taxonomy
        taxonomy_str = json.dumps(existing_taxonomy, indent=2)
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """You are an assistant that organizes knowledge by categorizing notes.
                    Analyze the note content and suggest relevant categories. Consider both existing categories
                    in the taxonomy and new categories that might be appropriate. Each note can have 1-3 categories."""
                },
                {
                    "role": "user",
                    "content": f"""Categorize this note:

Title: {note_title}

Content excerpt: {note_content[:1000]}

Existing categories in this note: {existing_categories}

Existing taxonomy: {taxonomy_str}

Please suggest 1-3 categories for this note. Categories should be broad enough to group related notes,
but specific enough to be meaningful. They should reflect the primary topics, domains, or themes of the note.

Return a JSON object with:
1. "categories": an array of 1-3 category strings
2. "explanation": a brief explanation of why these categories were chosen
3. "taxonomy_updates": any suggested additions or changes to the taxonomy

Example:
{{
  "categories": ["Artificial Intelligence", "Machine Learning"],
  "explanation": "This note discusses AI techniques including machine learning algorithms.",
  "taxonomy_updates": {{
    "Artificial Intelligence": {{
      "description": "The field of AI including various subfields and applications",
      "related_categories": ["Machine Learning", "Neural Networks"]
    }}
  }}
}}"""
                }
            ]
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        
        return result
        
    except Exception as e:
        print(f"Error categorizing note: {str(e)}")
        return {"categories": [], "explanation": "", "taxonomy_updates": {}}

def update_note_with_categories(note_path, note_content, categories):
    """Update a note with category metadata."""
    try:
        # Create the category metadata string
        category_str = ", ".join([f'"{c}"' for c in categories])
        metadata = f"categories: [{category_str}]"
        
        # Check if note already has category metadata
        if re.search(r'categories:\s*\[.*?\]', note_content):
            # Replace existing category metadata
            updated_content = re.sub(
                r'categories:\s*\[.*?\]', 
                metadata, 
                note_content
            )
        else:
            # Add new category metadata at the beginning of the file, after frontmatter if it exists
            if note_content.startswith("---"):
                # Find the end of the frontmatter
                fm_end = note_content.find("---", 3)
                if fm_end != -1:
                    # Insert categories before the end of frontmatter
                    updated_content = note_content[:fm_end] + metadata + "\n" + note_content[fm_end:]
                else:
                    # No proper frontmatter ending, add categories at the beginning
                    updated_content = "---\n" + metadata + "\n---\n" + note_content
            else:
                # No frontmatter, add categories at the beginning
                updated_content = "---\n" + metadata + "\n---\n" + note_content
        
        # Save the updated note
        with open(note_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        
        return True
        
    except Exception as e:
        print(f"Error updating note with categories: {str(e)}")
        return False

def update_taxonomy(taxonomy, updates):
    """Update the taxonomy with new categories or changes."""
    for category, data in updates.items():
        if category not in taxonomy:
            # Add new category
            taxonomy[category] = data
        else:
            # Update existing category
            if "description" in data:
                taxonomy[category]["description"] = data["description"]
            if "related_categories" in data:
                # Merge related categories
                existing_related = set(taxonomy[category].get("related_categories", []))
                new_related = set(data["related_categories"])
                taxonomy[category]["related_categories"] = list(existing_related.union(new_related))
    
    return taxonomy

def categorize_notes(notes):
    """Categorize multiple notes and update them with categories."""
    print(f"Categorizing {len(notes)} out of {len(notes)} total notes...")
    
    # Load or create the taxonomy - with timeout handling
    taxonomy_path = "category_taxonomy.json"
    try:
        print("Loading or creating category taxonomy...")
        taxonomy = load_or_create_category_taxonomy(taxonomy_path)
        print("Taxonomy loaded successfully")
    except Exception as e:
        print(f"Error in taxonomy loading: {str(e)}")
        # Create an empty taxonomy to continue processing
        taxonomy = {}
    
    updated_count = 0
    skipped_count = 0
    
    # Process notes in batches to provide progress feedback
    total_notes = len(notes)
    progress_interval = max(1, total_notes // 20)  # Show progress every 5%
    
    print(f"Beginning processing of {total_notes} notes...")
    for i, (path, note_data) in enumerate(notes.items()):
        # Show progress every 5% of notes
        if i % progress_interval == 0:
            print(f"Progress: {i}/{total_notes} notes processed ({i/total_notes*100:.1f}%)")
            
        try:
            content = note_data["content"] if isinstance(note_data, dict) else note_data
            title = os.path.splitext(os.path.basename(path))[0]
            
            # Get category suggestions
            try:
                print(f"Categorizing note: {title}...")
                result = categorize_note(content, title, taxonomy)
                print(f"Got categories for '{title}': {result.get('categories', [])}")
            except Exception as e:
                print(f"Error getting categories for {title}: {str(e)}")
                skipped_count += 1
                continue
            
            if not result["categories"]:
                print(f"No categories suggested for {title}")
                skipped_count += 1
                continue
            
            # Update the note with categories
            try:
                if update_note_with_categories(path, content, result["categories"]):
                    updated_count += 1
                    print(f"Updated categories for {title}")
                    
                    # Update the note content in the dictionary
                    if isinstance(note_data, dict):
                        # Re-read the file to get updated content
                        with open(path, "r", encoding="utf-8") as f:
                            note_data["content"] = f.read()
                    else:
                        # Re-read the file to get updated content
                        with open(path, "r", encoding="utf-8") as f:
                            notes[path] = f.read()
                    
                    # Update the taxonomy
                    if result["taxonomy_updates"]:
                        taxonomy = update_taxonomy(taxonomy, result["taxonomy_updates"])
                        print(f"Updated taxonomy with entries from {title}")
                else:
                    print(f"Failed to update categories for {title}")
                    skipped_count += 1
            except Exception as e:
                print(f"Error updating note {title}: {str(e)}")
                skipped_count += 1
                
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            skipped_count += 1
    
    # Save the updated taxonomy
    try:
        print("Saving updated category taxonomy...")
        save_category_taxonomy(taxonomy, taxonomy_path)
        print("Taxonomy saved successfully")
    except Exception as e:
        print(f"Error saving taxonomy: {str(e)}")
    
    print(f"Categorization completed: {updated_count} notes updated, {skipped_count} skipped")
    return updated_count

def cleanup_before_exit():
    """Clean up resources before exiting."""
    print("Performing cleanup before exit...")
    print("Note categorization tool interrupted. Some files may have been modified.")
    print("Cleanup completed. Goodbye!")

def print_obsidian_setup_instructions():
    """Print instructions for setting up Obsidian to use categories for graph visualization."""
    print("\n===== Obsidian Setup Instructions for Categories =====")
    print("To visualize categories in your Obsidian graph view:")
    print("1. Open Obsidian Settings > Appearance > CSS Snippets")
    print("2. Create a new CSS snippet file (e.g., 'category-colors.css')")
    print("3. Add CSS rules for each category to color nodes in the graph")
    print("4. Enable the CSS snippet in Obsidian")
    print("\nExample CSS snippet:")
    print("""
    /* Category Colors for Graph View */
    .graph-view.color-fill-tag[data-tag="categories/Programming"] {
        color: #e74c3c;
        background-color: rgba(231, 76, 60, 0.2);
    }
    
    .graph-view.color-fill-tag[data-tag="categories/Philosophy"] {
        color: #3498db;
        background-color: rgba(52, 152, 219, 0.2);
    }
    
    .graph-view.color-fill-tag[data-tag="categories/History"] {
        color: #2ecc71;
        background-color: rgba(46, 204, 113, 0.2);
    }
    """)
    print("\n5. In Obsidian Graph View, configure display settings to:")
    print("   - Show tags")
    print("   - Filter for 'categories/' tags")
    print("\nThis will color-code notes in your graph based on their categories.")

def main():
    """Main function to categorize notes."""
    # Set up clean interrupt handling
    signal_handler.setup_interrupt_handling()
    
    # Register cleanup function
    signal_handler.register_cleanup_function(cleanup_before_exit)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Categorize notes in an Obsidian vault")
    parser.add_argument("--vault-path", help="Path to Obsidian vault")
    parser.add_argument("--taxonomy-path", default="category_taxonomy.json", 
                        help="Path to store category taxonomy")
    parser.add_argument("--sample-size", type=int, default=5,
                        help="Number of random notes to process (default: 5)")
    args = parser.parse_args()
    
    vault_path = args.vault_path or os.getenv("OBSIDIAN_VAULT_PATH")
    if not vault_path:
        print("Error: OBSIDIAN_VAULT_PATH not set in environment or .env file")
        sys.exit(1)
    
    taxonomy_path = args.taxonomy_path
    sample_size = args.sample_size
    
    print(f"Loading notes from vault: {vault_path}")
    notes = load_notes(vault_path)
    
    print(f"Loading category taxonomy from: {taxonomy_path}")
    taxonomy = load_or_create_category_taxonomy(taxonomy_path)
    
    # Choose a random subset of notes to process
    num_notes = min(sample_size, len(notes))
    note_paths = random.sample(list(notes.keys()), num_notes)
    
    print(f"Processing {num_notes} random notes for categorization")
    updated = 0
    skipped = 0
    
    for path in note_paths:
        try:
            note = notes[path]
            content = note["content"]
            title = os.path.splitext(note["filename"])[0]
            
            print(f"Categorizing: {title}")
            
            # Get category suggestions
            result = categorize_note(content, title, taxonomy)
            
            if not result["categories"]:
                print(f"No categories suggested for {title}")
                skipped += 1
                continue
            
            # Update the note with categories
            if update_note_with_categories(path, content, result["categories"]):
                print(f"Updated {title} with categories: {', '.join(result['categories'])}")
                updated += 1
                
                # Update the taxonomy
                if result["taxonomy_updates"]:
                    taxonomy = update_taxonomy(taxonomy, result["taxonomy_updates"])
            else:
                print(f"Failed to update {title}")
                skipped += 1
                
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            skipped += 1
    
    # Save the updated taxonomy
    save_category_taxonomy(taxonomy, taxonomy_path)
    
    print(f"Categorization completed: {updated} notes updated, {skipped} skipped")
    return updated

if __name__ == "__main__":
    main()
