#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
note_categorizer.py - Categorize Obsidian notes with color tags for graph visualization

This script reads all markdown notes in an Obsidian vault and:
1. Extracts tags from all notes
2. Groups notes into categories based on tags
3. Adds a special CSS color class for graph visualization

Features:
- Categorizes notes based on their primary tag
- Creates visually distinct clusters in the Obsidian graph view
- Preserves existing links and content

Author: Jonathan Care <jonc@lacunae.org>
"""

import os
import sys
import re
from collections import defaultdict
from dotenv import load_dotenv
import utils

# Load environment variables from .env file
load_dotenv()

# Define categories and their colors
CATEGORIES = {
    "work": "#ff5555",        # Red
    "personal": "#50fa7b",    # Green
    "project": "#8be9fd",     # Cyan
    "idea": "#ffb86c",        # Orange
    "research": "#bd93f9",    # Purple
    "reference": "#f1fa8c",   # Yellow
    "journal": "#ff79c6",     # Pink
    "meeting": "#6272a4",     # Muted Blue
    "task": "#8be9fd",        # Light Blue
    "default": "#f8f8f2"      # White/Default
}

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

def determine_category(tags):
    """Determine the primary category of a note based on its tags."""
    # First check if any tag directly matches a category
    for tag in tags:
        # Remove the # prefix if present
        tag_name = tag[1:] if tag.startswith("#") else tag
        tag_name = tag_name.lower()
        
        if tag_name in CATEGORIES:
            return tag_name
        
        # Check for compound tags like "work/meeting"
        for category in CATEGORIES:
            if tag_name.startswith(f"{category}/") or tag_name.endswith(f"/{category}"):
                return category
    
    # If no direct match, check for broader categories
    for tag in tags:
        tag_name = tag[1:] if tag.startswith("#") else tag
        tag_name = tag_name.lower()
        
        for category in CATEGORIES:
            if category in tag_name:
                return category
    
    # Default category if no matches
    return "default"

def categorize_notes(notes):
    """Categorize all notes based on their tags."""
    categorized = 0
    
    for path, content in notes.items():
        # Extract tags
        tags = utils.extract_existing_tags(content)
        
        if not tags:
            continue
        
        # Determine the category
        category = determine_category(tags)
        
        # Add or update the cssclass in the YAML frontmatter
        if not re.search(r'^---\s*\n', content):
            # If no frontmatter exists, create it
            notes[path] = f"---\ncssclass: {category}\n---\n\n{content}"
        elif re.search(r'^---\s*\n.*?cssclass:.*?\n', content, re.DOTALL):
            # If cssclass already exists, update it
            notes[path] = re.sub(
                r'(cssclass:).*?(\n)',
                f'\\1 {category}\\2',
                content,
                count=1,
                flags=re.DOTALL
            )
        else:
            # If frontmatter exists but no cssclass, add it
            notes[path] = re.sub(
                r'^(---\s*\n)',
                f'\\1cssclass: {category}\n',
                content,
                count=1
            )
        
        categorized += 1
    
    print(f"Categorized {categorized} notes")
    return categorized

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

def print_obsidian_setup_instructions():
    """Print instructions for setting up CSS in Obsidian."""
    print("\nObsidian Setup Instructions:")
    print("1. Create a new CSS snippet in your vault:")
    print("   - Open Obsidian settings")
    print("   - Go to 'Appearance' → 'CSS snippets'")
    print("   - Click the folder icon to open the snippets folder")
    print("   - Create a new file called 'note-colors.css'")
    print("\n2. Add the following CSS to the file:")
    
    css = """/* Note colors for graph view */\n"""
    for category, color in CATEGORIES.items():
        css += f""".{category} {{
  --graph-color: {color};
  --graph-fill: {color}20;
  --graph-line: {color};
}}
"""
    print(f"\n```css\n{css}```\n")
    
    print("3. Enable the snippet in Obsidian settings")
    print("4. Open the graph view to see your categorized notes")

def main():
    """Main function to run note categorization."""
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
    if not vault_path:
        print("Error: OBSIDIAN_VAULT_PATH not set in environment or .env file")
        sys.exit(1)
    
    print(f"Loading notes from vault: {vault_path}")
    notes = load_notes(vault_path)
    
    print("Categorizing notes based on tags")
    categorized = categorize_notes(notes)
    
    print("Saving notes")
    saved = save_notes(notes, vault_path)
    
    if saved > 0:
        print(f"Categorized and saved {saved} notes")
        print_obsidian_setup_instructions()
    
    return saved

if __name__ == "__main__":
    main() 