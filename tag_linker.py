#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tag_linker.py - Create links between notes that share the same tags

This script reads all markdown notes in an Obsidian vault and:
1. Extracts tags from all notes
2. Identifies notes that share the same tags
3. For each note, adds a "Related Notes (by Tag)" section with links to related notes

Features:
- Configurable minimum tag match (default: at least 2 matching tags)
- Preserves existing links and sections

Author: Jonathan Care <jonc@lacunae.org>
"""

import os
import sys
import re
from collections import defaultdict
from dotenv import load_dotenv
import utils
import signal_handler

# Load environment variables from .env file
load_dotenv()

# Minimum number of matching tags to create a link
MIN_TAG_MATCH = 2

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

def extract_tags(notes):
    """Extract tags from all notes."""
    note_tags = {}  # Dict to store tags for each note
    tag_to_notes = defaultdict(list)  # Dict to store notes for each tag
    
    for path, note in notes.items():
        # Get content from note
        content = note["content"]
        
        # Extract tags from the content using utility function
        tags = utils.extract_existing_tags(content)
            
        # Store tags for this note
        note_tags[path] = tags
        
        # Store notes for each tag
        for tag in tags:
            # Remove the # prefix when storing in tag_to_notes
            tag_clean = tag[1:] if tag.startswith("#") else tag
            tag_to_notes[tag_clean].append(path)
    
    return note_tags, tag_to_notes

def build_relations(notes, note_tags, tag_to_notes):
    """Build relations between notes based on shared tags."""
    relations = {}
    
    # For each note, find other notes with matching tags
    for path, tags in note_tags.items():
        # Skip notes with no tags
        if not tags:
            continue
        
        # Track related notes and the tags they share
        related_notes = defaultdict(set)
        
        # For each tag, find other notes with the same tag
        for tag in tags:
            tag_clean = tag[1:] if tag.startswith("#") else tag
            for related_path in tag_to_notes[tag_clean]:
                if related_path != path:  # Don't link to self
                    related_notes[related_path].add(tag_clean)
        
        # Filter to notes with at least MIN_TAG_MATCH matching tags
        strong_relations = {
            rel_path: list(rel_tags)
            for rel_path, rel_tags in related_notes.items()
            if len(rel_tags) >= MIN_TAG_MATCH
        }
        
        # Store relations for this note
        if strong_relations:
            relations[path] = strong_relations
    
    return relations

def update_notes_with_relations(notes, relations, existing_links=None):
    """Update notes with links to related notes by tag."""
    updated = 0
    
    for path, related in relations.items():
        try:
            content = notes[path]["content"]
            
            # Extract existing links to avoid duplicate linking
            current_links = existing_links.get(path, []) if existing_links else utils.extract_existing_links(content)
            
            # Build the new section with links to related notes
            link_entries = []
            
            # Extract existing section content if it exists
            section_text, _ = utils.extract_section(content, "## Related Notes (by Tag)")
            existing_link_entries = []
            if section_text:
                existing_link_entries = section_text.split("\n")
            
            # Create entries for related notes
            for rel_path, shared_tags in related.items():
                # Get the note name from the related path
                rel_file = os.path.basename(rel_path)
                rel_note_name = os.path.splitext(rel_file)[0]
                
                # Skip if already linked in the document
                if rel_note_name in current_links:
                    continue
                
                # Add to current links to avoid duplicates in future iterations
                current_links.append(rel_note_name)
                
                # Format the link with shared tags
                tags_text = ", ".join([f"#{tag}" for tag in sorted(shared_tags)])
                link_entry = f"- [[{rel_note_name}]] - Shared tags: {tags_text}"
                link_entries.append(link_entry)
            
            # If we have no entries to add and no existing entries, skip
            if not link_entries and not existing_link_entries:
                continue
            
            # Merge existing and new link entries
            all_link_entries = utils.merge_links(existing_link_entries, link_entries)
            
            # Update the section in the content
            updated_content = utils.replace_section(
                content, 
                "## Related Notes (by Tag)", 
                "\n".join(all_link_entries)
            )
            
            # Save the updated content
            notes[path]["content"] = updated_content
            updated += 1
            
        except Exception as e:
            print(f"Error updating related notes for {path}: {str(e)}")
    
    return updated

def save_notes(notes, vault_path=None):
    """Save updated notes to disk."""
    saved = 0
    errors = 0
    
    for path, note in notes.items():
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(note["content"])
            saved += 1
        except Exception as e:
            print(f"Error saving {path}: {str(e)}")
            errors += 1
    
    print(f"Saved {saved} notes with {errors} errors")
    return saved

def cleanup_before_exit():
    """Clean up resources before exiting."""
    print("Performing cleanup before exit...")
    print("Tag linking tool interrupted. No files have been modified.")
    print("Cleanup completed. Goodbye!")

def main():
    """Main function to run tag-based linking."""
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
    
    print("Extracting tags from notes")
    note_tags, tag_to_notes = extract_tags(notes)
    
    print("Building relations between notes")
    relations = build_relations(notes, note_tags, tag_to_notes)
    
    print("Updating notes with related links")
    updated = update_notes_with_relations(notes, relations)
    
    print("Saving notes")
    saved = save_notes(notes)
    
    print(f"Added tag-based links to {updated} notes ({saved} saved)")
    return saved

if __name__ == "__main__":
    main() 