#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
note_categorizer.py - Automatically categorizes and colors Obsidian notes in the graph view

This script analyzes each note in an Obsidian vault using OpenAI, determines the most 
appropriate category for the note, and adds special color tags that can be used to 
visually distinguish different types of notes in Obsidian's graph view.

Author: Jonathan Care <jonc@lacunae.org>
"""

import os
import re
import json
import time
import hashlib
import tiktoken
import backoff
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "/Users/jonc/Obsidian/Jonathans Brain")
MODEL = "gpt-3.5-turbo"  # Using a less expensive model for categorization
MAX_TOKENS = 800  # Maximum tokens for note content analysis
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache_categories")

# Define categories and their associated color tags
CATEGORIES = {
    "concept": "#category-concept",
    "person": "#category-person",
    "project": "#category-project",
    "research": "#category-research",
    "reference": "#category-reference",
    "journal": "#category-journal",
    "note": "#category-note",
    "tool": "#category-tool",
    "event": "#category-event",
    "place": "#category-place"
}

# Create cache directory if it doesn't exist
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Initialize tokenizer for counting tokens
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    """Count the number of tokens in a text string."""
    return len(tokenizer.encode(text))

def truncate_text(text, max_tokens):
    """Truncate text to fit within max_tokens."""
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    # Keep the beginning and end of the text
    half_length = max_tokens // 2
    beginning = tokenizer.decode(tokens[:half_length])
    end = tokenizer.decode(tokens[-half_length:])
    
    return beginning + "\n\n[...content truncated...]\n\n" + end

def load_notes(vault_path):
    """Load all markdown notes from the vault."""
    notes = {}
    count = 0
    skipped = 0
    
    print(f"Loading notes from {vault_path}")
    for root, dirs, files in os.walk(vault_path):
        # Skip venv directory
        dirs[:] = [d for d in dirs if d != "venv"]
        
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                        notes[path] = content
                        count += 1
                except Exception as e:
                    print(f"Error reading file {path}: {str(e)}")
                    skipped += 1
    
    print(f"Loaded {count} notes, skipped {skipped} due to errors")
    return notes

def generate_cache_key(content):
    """Generate a unique cache key based on note content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def get_category_from_cache(cache_key):
    """Retrieve category from cache if it exists."""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading cache file: {str(e)}")
    return None

def save_category_to_cache(cache_key, category_data):
    """Save category to cache."""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        with open(cache_file, 'w') as f:
            json.dump(category_data, f)
    except Exception as e:
        print(f"Error writing to cache file: {str(e)}")

@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    max_time=60,
    giveup=lambda e: "invalid_request_error" in str(e),
)
def determine_category(content, filename):
    """Determine the most appropriate category for a note using OpenAI."""
    # Truncate content to fit within token limits
    truncated_content = truncate_text(content, MAX_TOKENS)
    
    # Create a prompt for the model
    categories_str = ", ".join(CATEGORIES.keys())
    prompt = f"""Analyze this Obsidian note and determine the most appropriate category for it. 
Filename: {filename}

Content: 
{truncated_content}

Choose exactly ONE category from this list: {categories_str}

Provide your answer in this exact format:
CATEGORY: [chosen_category]
REASON: [brief explanation]
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        
        result = response.choices[0].message.content.strip()
        
        # Extract category from response
        category_match = re.search(r'CATEGORY:\s*(\w+)', result)
        reason_match = re.search(r'REASON:\s*(.+?)$', result, re.DOTALL)
        
        if category_match:
            category = category_match.group(1).lower()
            reason = reason_match.group(1).strip() if reason_match else "No reason provided"
            
            # Ensure category is valid
            if category not in CATEGORIES:
                print(f"Invalid category '{category}' returned for {filename}, defaulting to 'note'")
                category = "note"
                
            return {
                "category": category,
                "reason": reason,
                "color_tag": CATEGORIES[category]
            }
        else:
            print(f"Failed to extract category from response for {filename}")
            return {
                "category": "note",
                "reason": "Failed to determine category",
                "color_tag": CATEGORIES["note"]
            }
            
    except Exception as e:
        print(f"Error determining category for {filename}: {str(e)}")
        return {
            "category": "note",
            "reason": f"Error: {str(e)}",
            "color_tag": CATEGORIES["note"]
        }

def update_note_with_category(content, category_data):
    """Add or update category tags in the note content."""
    category_tag = category_data["color_tag"]
    
    # Check if note has existing category tags
    existing_category_tags = []
    for tag in CATEGORIES.values():
        if tag in content:
            existing_category_tags.append(tag)
    
    # Replace existing category tags if any
    for tag in existing_category_tags:
        content = content.replace(tag, "")
    
    # Add the new category tag to the tags section if it exists
    if "#tags:" in content:
        # Add to existing tags section
        content = re.sub(
            r'(#tags:[^\n]*)',
            r'\1 ' + category_tag,
            content
        )
    else:
        # Create a new tags section at the top
        content = f"#tags: {category_tag}\n\n{content}"
    
    return content

def categorize_notes(notes):
    """Categorize all notes and add appropriate tags."""
    updated = 0
    skipped = 0
    categories_count = {category: 0 for category in CATEGORIES}
    
    print("Categorizing notes...")
    for path, content in tqdm(notes.items()):
        try:
            filename = os.path.basename(path)
            cache_key = generate_cache_key(content)
            cached_category = get_category_from_cache(cache_key)
            
            if cached_category:
                category_data = cached_category
            else:
                # Determine category using AI
                category_data = determine_category(content, filename)
                save_category_to_cache(cache_key, category_data)
                time.sleep(0.1)  # Small delay to avoid rate limiting
            
            # Update the note with category tag
            notes[path] = update_note_with_category(content, category_data)
            updated += 1
            categories_count[category_data["category"]] += 1
            
        except Exception as e:
            print(f"Error categorizing {path}: {str(e)}")
            skipped += 1
    
    print(f"Categorized {updated} notes, skipped {skipped} due to errors")
    print("Category distribution:")
    for category, count in categories_count.items():
        if count > 0:
            print(f"  - {category}: {count} notes")
    
    return updated

def save_notes(notes, vault_path):
    """Save the updated notes back to disk."""
    saved = 0
    failed = 0
    
    print("Saving categorized notes...")
    for path, content in tqdm(notes.items()):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
                saved += 1
        except Exception as e:
            print(f"Error writing to file {path}: {str(e)}")
            failed += 1
    
    print(f"Saved {saved} notes, failed to save {failed} notes")
    return saved

def print_obsidian_setup_instructions():
    """Print instructions for setting up Obsidian graph view colors."""
    print("\n--- Obsidian Graph View Setup Instructions ---")
    print("To see your categorized notes in different colors:")
    print("1. Open Obsidian and go to your vault")
    print("2. Click on the Graph View button in the left sidebar")
    print("3. Click on the settings icon (gear) in the graph view")
    print("4. In the 'Groups' section, create a group for each category:")
    
    for category, tag in CATEGORIES.items():
        print(f"   - Add a group for '{category}': Enter '{tag}' in the search field")
        print(f"     and choose a color for {category} notes")
        
    print("\nYour notes should now be colored based on their categories in the Graph View!")

if __name__ == "__main__":
    try:
        print(f"Using vault path: {VAULT_PATH}")
        print(f"Using model: {MODEL}")
        
        # Step 1: Load all notes
        notes = load_notes(VAULT_PATH)
        if not notes:
            print("No notes found! Check the vault path.")
            exit(1)
        
        # Step 2: Categorize notes
        categorized = categorize_notes(notes)
        
        # Step 3: Save the updated notes
        saved = save_notes(notes, VAULT_PATH)
        
        if saved > 0:
            print(f"✅ Note categorization completed! Added category tags to {saved} notes.")
            print_obsidian_setup_instructions()
        else:
            print("❌ No notes were saved. Check logs for errors.")
            
    except Exception as e:
        import traceback
        print(f"❌ Error during execution: {str(e)}")
        traceback.print_exc() 