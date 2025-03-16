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

load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "/Users/jonc/Obsidian/Jonathans Brain")
print(f"Using vault path: {VAULT_PATH}")

TAG_PROMPT = """
You are an AI assistant tasked with assigning 3-5 concise, relevant hashtags to the following note. Provide tags in lowercase without spaces (e.g., #projectmanagement, #python).

Note:
{}

Tags:
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

def generate_tags_for_note(note):
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "user", "content": TAG_PROMPT.format(note[:2000])}
            ],
            temperature=0.2,
        )
        tags = response.choices[0].message.content.strip()
        # Ensure tags have the # prefix
        tags = ' '.join([tag if tag.startswith('#') else f'#{tag}' for tag in tags.split()])
        return tags
    except Exception as e:
        print(f"Error generating tags: {str(e)}")
        return "#error"

def insert_tags(notes):
    updated = 0
    skipped = 0
    
    for path in tqdm(notes, desc="Generating tags"):
        content = notes[path]
        file_name = os.path.basename(path)
        
        try:
            if "#tags:" not in content.lower():
                print(f"Adding new tags to {file_name}")
                tags = generate_tags_for_note(content)
                tag_section = f"\n\n#tags: {tags}\n"
                notes[path] = content + tag_section
                updated += 1
            else:
                print(f"Updating existing tags in {file_name}")
                tags = generate_tags_for_note(content)
                notes[path] = re.sub(r"#tags:.*", f"#tags: {tags}", content, flags=re.DOTALL)
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
