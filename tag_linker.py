#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tag_linker.py - Links Obsidian notes based on shared tags

This script analyzes all notes in an Obsidian vault, extracts tags from each note,
and creates "Related Notes (by Tag)" sections based on the number of shared tags.
It helps organize notes by creating connections between content with similar tags.

Author: Jonathan Care <jonc@lacunae.org>
"""

import os
import re
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "/Users/jonc/Obsidian/Jonathans Brain")
# Minimum number of shared tags required to create a link between notes
MIN_SHARED_TAGS = 2
# Maximum number of related notes to show per note
MAX_RELATED_NOTES = 10

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
                        # Store both full path and filename
                        notes[path] = {
                            "content": content,
                            "filename": file
                        }
                        count += 1
                except Exception as e:
                    print(f"Error reading file {path}: {str(e)}")
                    skipped += 1
    
    print(f"Loaded {count} notes, skipped {skipped} due to errors")
    return notes

def extract_tags(notes):
    """Extract tags from all notes."""
    note_tags = {}
    tag_to_notes = defaultdict(list)
    
    print("Extracting tags from notes...")
    for path, note_data in tqdm(notes.items()):
        content = note_data["content"]
        
        # Find all tags in the content
        tags = []
        # Look for #tags: section first
        tags_section_match = re.search(r'#tags:\s*(.*?)(\n\n|\n$|$)', content, re.IGNORECASE | re.DOTALL)
        if tags_section_match:
            tags_text = tags_section_match.group(1).strip()
            # Extract tags from the tags section
            tags = [tag.strip() for tag in re.findall(r'#\w+', tags_text)]
        
        # Also find other inline tags in the document
        inline_tags = re.findall(r'#([a-zA-Z0-9_]+)', content)
        for tag in inline_tags:
            if f"#{tag}" not in tags:  # Avoid duplicates
                tags.append(f"#{tag}")
        
        if tags:
            note_tags[path] = tags
            # Build the reverse index: tag -> notes
            for tag in tags:
                tag_to_notes[tag].append(path)
    
    print(f"Extracted tags from {len(note_tags)} notes, found {len(tag_to_notes)} unique tags")
    return note_tags, tag_to_notes

def build_relations(notes, note_tags, tag_to_notes):
    """Build relations between notes based on shared tags."""
    relations = defaultdict(list)
    
    print("Building relations between notes...")
    for path, tags in tqdm(note_tags.items()):
        related_notes = defaultdict(int)
        
        # Find notes that share tags with this note
        for tag in tags:
            for related_path in tag_to_notes[tag]:
                if related_path != path:  # Don't relate a note to itself
                    related_notes[related_path] += 1
        
        # Filter notes that share at least MIN_SHARED_TAGS
        filtered_relations = [(p, count) for p, count in related_notes.items() 
                              if count >= MIN_SHARED_TAGS]
        
        # Sort by number of shared tags, descending
        sorted_relations = sorted(filtered_relations, key=lambda x: x[1], reverse=True)
        
        # Limit to MAX_RELATED_NOTES
        relations[path] = sorted_relations[:MAX_RELATED_NOTES]
    
    return relations

def update_notes_with_relations(notes, relations):
    """Update notes with related notes sections."""
    updated = 0
    skipped = 0
    
    print("Updating notes with relations...")
    for path, related_paths in tqdm(relations.items()):
        if not related_paths:
            continue
        
        try:
            content = notes[path]["content"]
            
            # Create links section
            links = []
            for related_path, count in related_paths:
                related_filename = notes[related_path]["filename"]
                # Create the Obsidian link using just the filename without extension
                note_name = os.path.splitext(related_filename)[0]
                links.append(f"- [[{note_name}]] ({count} shared tags)")
            
            link_section = "\n\n## Related Notes (by Tag)\n" + "\n".join(links)
            
            # Check if the note already has a related notes section
            if "## Related Notes (by Tag)" in content:
                # Replace existing section
                content = re.sub(
                    r"## Related Notes \(by Tag\).*?(?=\n## |\n#|\Z)", 
                    f"## Related Notes (by Tag)\n{chr(10).join(links)}\n\n", 
                    content, 
                    flags=re.DOTALL
                )
            else:
                # Add new section
                content += link_section
            
            # Update the note content
            notes[path]["content"] = content
            updated += 1
            
        except Exception as e:
            print(f"Error updating {path}: {str(e)}")
            skipped += 1
    
    print(f"Updated {updated} notes, skipped {skipped} due to errors")
    return updated

def save_notes(notes):
    """Save the updated notes back to disk."""
    saved = 0
    failed = 0
    
    print("Saving updated notes...")
    for path, note_data in tqdm(notes.items()):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(note_data["content"])
                saved += 1
        except Exception as e:
            print(f"Error writing to file {path}: {str(e)}")
            failed += 1
    
    print(f"Saved {saved} notes, failed to save {failed} notes")
    return saved

if __name__ == "__main__":
    try:
        print(f"Using vault path: {VAULT_PATH}")
        print(f"Minimum shared tags: {MIN_SHARED_TAGS}")
        print(f"Maximum related notes: {MAX_RELATED_NOTES}")
        
        # Step 1: Load all notes
        notes = load_notes(VAULT_PATH)
        if not notes:
            print("No notes found! Check the vault path.")
            exit(1)
        
        # Step 2: Extract tags from all notes
        note_tags, tag_to_notes = extract_tags(notes)
        if not note_tags:
            print("No tags found in any notes!")
            exit(1)
        
        # Step 3: Build relations based on shared tags
        relations = build_relations(notes, note_tags, tag_to_notes)
        
        # Step 4: Update notes with related notes sections
        updated = update_notes_with_relations(notes, relations)
        
        # Step 5: Save the updated notes
        saved = save_notes(notes)
        
        if saved > 0:
            print(f"✅ Tag-based linking completed! Updated and saved {saved} notes.")
        else:
            print("❌ No notes were saved. Check file permissions.")
            
    except Exception as e:
        import traceback
        print(f"❌ Error during execution: {str(e)}")
        traceback.print_exc() 