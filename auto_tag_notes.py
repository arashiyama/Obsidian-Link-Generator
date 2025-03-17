#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_tag_notes.py - Automatically generates tags for Obsidian notes using OpenAI

This script reads each note in an Obsidian vault, uses the OpenAI API to generate
appropriate tags based on the content, and updates or adds a #tags section to each note.
Features include filtering directories, preserving existing tags, and debugging information.

Author: Jonathan Care <jonc@lacunae.org>
"""

import os, re
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import traceback
import utils

load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "/Users/jonc/Obsidian/Jonathans Brain")
print(f"Using vault path: {VAULT_PATH}")

TAG_PROMPT = """
You are an AI assistant tasked with assigning 3-5 concise, relevant hashtags to the following note. Provide tags in lowercase without spaces (e.g., #projectmanagement, #python).

The note already has these existing tags: {existing_tags}
Please suggest additional relevant tags that DON'T duplicate the existing ones.

Note:
{note_content}

New Tags (don't include existing ones):
"""

def load_notes(vault_path):
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
                        notes[path] = content  # Store full path as key instead of just filename
                        count += 1
                except Exception as e:
                    print(f"Error reading file {path}: {str(e)}")
                    skipped += 1
    
    print(f"Loaded {count} notes, skipped {skipped} due to errors")
    return notes

def generate_tags_for_note(note_content, existing_tags):
    """Generate new tags for a note, taking into account existing tags."""
    existing_tags_str = ", ".join(existing_tags) if existing_tags else "none"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "user", "content": TAG_PROMPT.format(
                    existing_tags=existing_tags_str,
                    note_content=note_content[:2000]
                )}
            ],
            temperature=0.2,
        )
        new_tags = response.choices[0].message.content.strip()
        
        # Ensure new tags have the # prefix and are unique
        formatted_tags = []
        for tag in new_tags.split():
            if not tag.startswith('#'):
                tag = f'#{tag}'
            formatted_tags.append(tag)
        
        # Deduplicate against existing tags
        return [tag for tag in formatted_tags 
                if not any(existing.lower() == tag.lower() for existing in existing_tags)]
    except Exception as e:
        print(f"Error generating tags: {str(e)}")
        return ["#error"]

def insert_tags(notes):
    updated = 0
    skipped = 0
    
    for path in tqdm(notes, desc="Generating tags"):
        content = notes[path]
        file_name = os.path.basename(path)
        
        try:
            # Extract existing tags using the utility function
            existing_tags = utils.extract_existing_tags(content)
            print(f"Found {len(existing_tags)} existing tags in {file_name}")
            
            # Generate new tags
            new_tags = generate_tags_for_note(content, existing_tags)
            if not new_tags:
                print(f"No new tags generated for {file_name}, keeping existing tags")
                continue
                
            print(f"Generated {len(new_tags)} new tags for {file_name}")
            
            # Combine all tags and deduplicate
            all_tags = utils.deduplicate_tags(existing_tags + new_tags)
            
            # Format all tags for the #tags section
            tags_text = " ".join(all_tags)
            
            if "#tags:" not in content.lower():
                print(f"Adding new tags section to {file_name}")
                tag_section = f"\n\n#tags: {tags_text}\n"
                notes[path] = content + tag_section
                updated += 1
            else:
                print(f"Updating existing tags section in {file_name}")
                notes[path] = re.sub(r"#tags:.*?(\n\n|\n$|$)", f"#tags: {tags_text}\n", content, flags=re.DOTALL)
                updated += 1
                
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            traceback.print_exc()
            skipped += 1
    
    print(f"Updated {updated} notes, skipped {skipped} due to errors")
    return updated

def save_notes(notes, vault_path):
    saved = 0
    failed = 0
    
    for path, content in notes.items():
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
                saved += 1
        except Exception as e:
            print(f"Error writing to file {path}: {str(e)}")
            failed += 1
    
    print(f"Saved {saved} notes, failed to save {failed} notes")
    return saved

if __name__ == "__main__":
    try:
        notes = load_notes(VAULT_PATH)
        if not notes:
            print("No notes found! Check the vault path.")
            exit(1)
            
        updated = insert_tags(notes)
        if updated == 0:
            print("No notes were updated. Check the tagging logic.")
            exit(1)
            
        saved = save_notes(notes, VAULT_PATH)
        if saved > 0:
            print(f"✅ Auto-tagging completed! Updated and saved {saved} notes.")
        else:
            print("❌ No notes were saved. Check file permissions.")
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        traceback.print_exc()
