#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genai_linker.py - Create links between notes using GenAI for relevance analysis

This script reads markdown notes from an Obsidian vault and:
1. Extracts titles and summaries from notes
2. Uses OpenAI GPT to find relationships between notes with explanations
3. Adds a "Related Notes (GenAI)" section with links and explanations

Features:
- Intelligent relevance scoring
- Natural language explanations for relationships
- Avoids duplicating existing links

Author: Jonathan Care <jonc@lacunae.org>
"""

import os
import sys
import re
import json
import random
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

def extract_titles_and_summaries(notes):
    """Extract titles and create summaries for each note."""
    summaries = {}
    
    for path, note in notes.items():
        # Handle different note formats
        if isinstance(note, dict) and "content" in note:
            content = note["content"]
            # Try to get filename from the note dictionary
            if "filename" in note:
                filename = note["filename"]
            else:
                # Extract filename from path
                filename = os.path.basename(path)
        else:
            # In case note is directly a string
            content = note
            filename = os.path.basename(path)
        
        # Extract title - first use H1 if available, else use filename
        title = os.path.splitext(filename)[0]
        h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if h1_match:
            title = h1_match.group(1).strip()
        
        # Create a summary from the first 500 characters, stopping at the nearest paragraph break
        summary = content[:500]
        last_para_break = summary.rfind("\n\n")
        if last_para_break > 100:  # Ensure we have at least 100 chars
            summary = summary[:last_para_break]
        
        # Store the title and summary
        summaries[path] = {
            "title": title,
            "summary": summary.strip()
        }
    
    return summaries

def find_relevant_notes(target_path, notes, summaries, max_notes=5):
    """Find relevant notes for a target note using OpenAI API."""
    target_note = notes[target_path]
    
    # Handle different note formats for target note
    if isinstance(target_note, dict) and "content" in target_note:
        target_content = target_note["content"]
    else:
        # In case target_note is directly a string
        target_content = target_note
    
    target_summary = summaries[target_path]
    
    # Get random sample of other notes (excluding the target)
    other_paths = [path for path in notes.keys() if path != target_path]
    
    # Limit to 20 random notes to keep API calls manageable
    if len(other_paths) > 20:
        other_paths = random.sample(other_paths, 20)
    
    # Create the list of other note summaries
    candidates = []
    for path in other_paths:
        candidates.append({
            "path": path,
            "title": summaries[path]["title"],
            "summary": summaries[path]["summary"]
        })
    
    # Skip if we have no candidates
    if not candidates:
        return []
    
    try:
        # Prepare the prompt for the API
        prompt = {
            "target_note": {
                "title": target_summary["title"],
                "content": target_content[:2000]  # Limit content to first 2000 chars
            },
            "candidate_notes": candidates
        }
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """You are an assistant that finds relationships between notes in a knowledge base.
                    Analyze the target note and the candidate notes, and identify which candidates are most
                    relevant to the target. Provide a score from 1-10 (10 being highly related) and a brief
                    explanation. Return a JSON array with the path and score for each relevant note."""
                },
                {
                    "role": "user",
                    "content": f"""Find the most relevant notes to this target note:

Target: {json.dumps(prompt["target_note"])}

Candidates: {json.dumps(prompt["candidate_notes"])}

For each candidate, evaluate how related it is to the target note. 
Choose at most {max_notes} notes that are most relevant.
Return a JSON array with "related_notes" containing:
- path (string)
- score (integer 1-10)
- reason (string, 1-2 sentences explaining the relationship)

Example:
{{
  "related_notes": [
    {{
      "path": "/path/to/note.md",
      "score": 8,
      "reason": "Both notes discuss similar concepts and reference related theories."
    }}
  ]
}}"""
                }
            ]
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        
        # Extract the related notes
        related_notes = result.get("related_notes", [])
        
        # Only include notes with a score of at least 5
        filtered_notes = [
            {"path": note["path"], "score": note["score"], "reason": note["reason"]}
            for note in related_notes
            if note["score"] >= 5
        ]
        
        return filtered_notes
        
    except Exception as e:
        print(f"Error finding relevant notes: {str(e)}")
        return []

def save_notes(notes):
    """Save updated notes to disk."""
    saved = 0
    errors = 0
    
    for path, note in notes.items():
        try:
            # Handle different note formats
            if isinstance(note, dict) and "content" in note:
                content = note["content"]
            else:
                # In case note is directly a string
                content = note
                
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
    print("GenAI linking tool interrupted. No files have been modified.")
    print("Cleanup completed. Goodbye!")

def main():
    """Main function to run GenAI linking."""
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
    
    print("Extracting titles and summaries")
    summaries = extract_titles_and_summaries(notes)
    
    # Choose a random subset of notes to process
    num_notes = min(10, len(notes))  # Process at most 10 notes for demonstration
    note_paths = random.sample(list(notes.keys()), num_notes)
    
    print(f"Processing {num_notes} random notes with GenAI linking")
    updated = 0
    skipped = 0
    
    for path in note_paths:
        try:
            # Extract existing links to avoid duplicate linking
            # Handle different note formats
            if isinstance(notes[path], dict) and "content" in notes[path]:
                content = notes[path]["content"]
            else:
                # In case the note is directly a string
                content = notes[path]
                # Ensure notes[path] is in the right format for later updates
                notes[path] = {"content": content}
                
            current_links = utils.extract_existing_links(content)
            
            # Find relevant notes
            relevant_notes = find_relevant_notes(path, notes, summaries)
            
            if not relevant_notes:
                continue
            
            # Extract existing GenAI related notes section if it exists
            section_text, _ = utils.extract_section(content, "## Related Notes (GenAI)")
            existing_link_entries = []
            if section_text:
                existing_link_entries = section_text.split("\n")
            
            # Create new link entries for relevant notes
            new_link_entries = []
            for rel_note in relevant_notes:
                related_path = rel_note["path"]
                # Handle both dictionary formats 
                if isinstance(notes[related_path], dict) and "filename" in notes[related_path]:
                    # Standard format from genai_linker.load_notes
                    related_filename = notes[related_path]["filename"]
                    note_name = os.path.splitext(related_filename)[0]
                else:
                    # Alternative format or directly from obsidian_enhance.py
                    # Extract filename from the path
                    note_name = os.path.splitext(os.path.basename(related_path))[0]
                
                # Skip if already linked in the document
                if note_name in current_links:
                    continue
                
                # Format the link with relevance score and reason
                link_entry = f"- [[{note_name}]] (Score: {rel_note['score']}/10)\n  - {rel_note['reason']}"
                new_link_entries.append(link_entry)
                
                # Add to current links to avoid duplicates in future iterations
                current_links.append(note_name)
            
            # If we have no entries to add and no existing entries, skip
            if not new_link_entries and not existing_link_entries:
                continue
            
            # Merge existing and new link entries
            all_link_entries = utils.merge_links(existing_link_entries, new_link_entries)
            
            # Update the section in the content
            updated_content = utils.replace_section(
                content, 
                "## Related Notes (GenAI)", 
                "\n".join(all_link_entries)
            )
            
            # Save the updated content
            notes[path]["content"] = updated_content
            updated += 1
            
        except Exception as e:
            print(f"Error updating {path}: {str(e)}")
            skipped += 1
    
    print("Saving notes")
    saved = save_notes(notes)
    
    print(f"Added GenAI links to {updated} notes ({saved} saved)")
    return saved

if __name__ == "__main__":
    main()
