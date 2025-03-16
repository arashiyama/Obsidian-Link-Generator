#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genai_linker.py - Creates insightful links between Obsidian notes using Generative AI

This script analyzes notes in an Obsidian vault and uses the OpenAI GPT-4 API to identify 
relevant relationships between notes with detailed explanations. It creates "Related Notes (GenAI)" 
sections that include relevance scores and explanations of the relationships.
Features include content truncation, caching, and selective processing.

Author: Jonathan Care <jonc@lacunae.org>
"""

import os
import re
import random
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import time
import tiktoken
import hashlib
import json

load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "/Users/jonc/Obsidian/Jonathans Brain")
MODEL = "gpt-4-turbo-preview"  # Use a powerful model for better reasoning
MAX_TOKENS = 1000  # Maximum tokens to use for note content in the prompt
MAX_RELATED_NOTES = 5  # Maximum number of related notes to identify per note
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache_genai")

# Create cache directory if it doesn't exist
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Initialize the tokenizer for counting tokens
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    """Count the number of tokens in a text string."""
    return len(tokenizer.encode(text))

def truncate_text(text, max_tokens):
    """Truncate text to fit within max_tokens."""
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    # Keep the beginning and end of the text, which often contain key information
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
                rel_path = os.path.relpath(path, vault_path)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Store both full path, filename, and content
                        note_name = os.path.splitext(file)[0]
                        notes[path] = {
                            "content": content,
                            "filename": file,
                            "name": note_name,
                            "rel_path": rel_path
                        }
                        count += 1
                except Exception as e:
                    print(f"Error reading file {path}: {str(e)}")
                    skipped += 1
    
    print(f"Loaded {count} notes, skipped {skipped} due to errors")
    return notes

def extract_titles_and_summaries(notes):
    """Extract titles and first paragraph as summaries for each note."""
    summaries = {}
    
    print("Extracting summaries...")
    for path, note_data in tqdm(notes.items()):
        content = note_data["content"]
        
        # Use filename as title
        title = note_data["name"]
        
        # Get first paragraph as summary
        lines = content.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]
        summary = ""
        if non_empty_lines:
            # Skip header lines that start with # and get first paragraph
            for line in non_empty_lines:
                if not line.startswith("#") and len(line) > 10:  # Skip short lines
                    summary = line
                    break
        
        # If no good summary found, use first 100 characters
        if not summary and content:
            summary = content[:100].replace("\n", " ")
        
        summaries[path] = {
            "title": title,
            "summary": summary[:200] + ("..." if len(summary) > 200 else "")
        }
    
    return summaries

def generate_cache_key(source_note, chunk_index, total_chunks):
    """Generate a unique cache key based on source note path and chunk info."""
    key_str = f"{source_note}_chunk{chunk_index}_of_{total_chunks}"
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()

def get_from_cache(cache_key):
    """Retrieve data from cache if it exists."""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading cache file: {str(e)}")
    return None

def save_to_cache(cache_key, data):
    """Save data to cache."""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error writing to cache file: {str(e)}")

def find_relevant_notes(source_path, notes, summaries, max_notes=MAX_RELATED_NOTES):
    """Find relevant notes for a given source note using GenAI."""
    source_data = notes[source_path]
    source_content = source_data["content"]
    source_title = source_data["name"]
    
    # Truncate source content to fit within token limits
    truncated_content = truncate_text(source_content, MAX_TOKENS)
    
    # We need to chunk the candidate notes since there might be too many
    # to fit in a single prompt
    candidate_paths = list(notes.keys())
    candidate_paths.remove(source_path)  # Remove source note from candidates
    
    # Shuffle candidates to get a different set for each chunk
    random.shuffle(candidate_paths)
    
    # Number of notes to process in each chunk
    chunk_size = 20
    total_chunks = (len(candidate_paths) + chunk_size - 1) // chunk_size
    
    # Process in smaller batches to avoid context length issues
    all_relevant_notes = []
    
    # Process small randomly selected chunks of notes
    for i in range(min(total_chunks, 5)):  # Limit to 5 chunks maximum
        chunk_start = i * chunk_size
        chunk_end = min((i + 1) * chunk_size, len(candidate_paths))
        chunk = candidate_paths[chunk_start:chunk_end]
        
        # Generate cache key for this chunk
        cache_key = generate_cache_key(source_path, i, total_chunks)
        cached_result = get_from_cache(cache_key)
        
        if cached_result:
            all_relevant_notes.extend(cached_result)
            continue
        
        # Build information for each candidate note
        candidates_info = []
        for path in chunk:
            candidates_info.append({
                "id": path,
                "title": summaries[path]["title"],
                "summary": summaries[path]["summary"]
            })
        
        # Skip if no candidates in this chunk
        if not candidates_info:
            continue
        
        # Create prompt for the LLM
        prompt = f"""You are an expert knowledge graph builder for a personal Obsidian knowledge base.

I'll provide you with a source note and a list of other notes. Your task is to identify which notes are most relevant to the source note and explain why.

SOURCE NOTE TITLE: {source_title}
SOURCE NOTE CONTENT:
{truncated_content}

CANDIDATE NOTES:
"""
        
        for idx, candidate in enumerate(candidates_info, 1):
            prompt += f"{idx}. TITLE: {candidate['title']}\n   SUMMARY: {candidate['summary']}\n\n"
        
        prompt += f"""
Identify up to {max_notes} notes that are most relevant to the source note. For each relevant note:
1. Explain specifically why it's relevant and how it connects to the source note
2. Rate the relevance on a scale of 1-10 where 10 is highest

Format your response as follows:
RELEVANT NOTE #X: [Note Title]
RELEVANCE SCORE: [1-10]
REASON: [Your detailed explanation of why this note is relevant]

Do not include notes that have little to no relevance to the source note. Focus on meaningful connections.
"""
        
        try:
            # Make API call
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse the results
            chunk_relevant_notes = []
            relevant_sections = re.split(r'RELEVANT NOTE #\d+:', result)[1:]
            
            for section in relevant_sections:
                # Extract title, score, and reason
                title_match = re.search(r'^\s*([^\n]+)', section)
                score_match = re.search(r'RELEVANCE SCORE:\s*(\d+)', section)
                reason_match = re.search(r'REASON:\s*(.+?)(?=\n\n|$)', section, re.DOTALL)
                
                if title_match and score_match and reason_match:
                    title = title_match.group(1).strip()
                    score = int(score_match.group(1))
                    reason = reason_match.group(1).strip()
                    
                    # Find the note ID based on title
                    note_id = None
                    for candidate in candidates_info:
                        if candidate["title"].lower() == title.lower() or title.lower() in candidate["title"].lower():
                            note_id = candidate["id"]
                            break
                    
                    if note_id:
                        chunk_relevant_notes.append({
                            "path": note_id,
                            "title": title,
                            "score": score,
                            "reason": reason
                        })
            
            # Cache the results
            save_to_cache(cache_key, chunk_relevant_notes)
            
            # Add to overall results
            all_relevant_notes.extend(chunk_relevant_notes)
            
            # Sleep to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error finding relevant notes for {source_title}: {str(e)}")
    
    # Sort by relevance score and limit to max_notes
    sorted_notes = sorted(all_relevant_notes, key=lambda x: x["score"], reverse=True)
    return sorted_notes[:max_notes]

def update_notes_with_relations(notes, find_related_func):
    """Update notes with related notes sections based on GenAI relevance."""
    updated = 0
    skipped = 0
    
    # First extract summaries for all notes
    summaries = extract_titles_and_summaries(notes)
    
    # Select a subset of notes to process if there are too many
    all_paths = list(notes.keys())
    if len(all_paths) > 100:
        # Process only a random sample to keep runtime reasonable
        print(f"Selecting 100 random notes out of {len(all_paths)} total notes to process")
        random.shuffle(all_paths)
        paths_to_process = all_paths[:100]
    else:
        paths_to_process = all_paths
    
    print(f"Finding related notes for {len(paths_to_process)} notes...")
    for path in tqdm(paths_to_process):
        try:
            # Find relevant notes
            relevant_notes = find_related_func(path, notes, summaries)
            
            if not relevant_notes:
                continue
            
            content = notes[path]["content"]
            
            # Create links section
            links = []
            for rel_note in relevant_notes:
                related_path = rel_note["path"]
                related_filename = notes[related_path]["filename"]
                note_name = os.path.splitext(related_filename)[0]
                
                # Format the link with relevance score and reason
                links.append(f"- [[{note_name}]] (Score: {rel_note['score']}/10)\n  - {rel_note['reason']}")
            
            link_section = "\n\n## Related Notes (GenAI)\n" + "\n".join(links)
            
            # Check if the note already has a GenAI related notes section
            if "## Related Notes (GenAI)" in content:
                # Replace existing section
                content = re.sub(
                    r"## Related Notes \(GenAI\).*?(?=\n## |\n#|\Z)", 
                    f"## Related Notes (GenAI)\n{chr(10).join(links)}\n\n", 
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
            traceback.print_exc()
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
        import traceback
        print(f"Using vault path: {VAULT_PATH}")
        print(f"Using model: {MODEL}")
        print(f"Maximum related notes per note: {MAX_RELATED_NOTES}")
        
        # Step 1: Load all notes
        notes = load_notes(VAULT_PATH)
        if not notes:
            print("No notes found! Check the vault path.")
            exit(1)
        
        # Step 2: Update notes with relations found by GenAI
        updated = update_notes_with_relations(notes, find_relevant_notes)
        
        # Step 3: Save the updated notes
        saved = save_notes(notes)
        
        if saved > 0:
            print(f"✅ GenAI-based linking completed! Updated and saved {saved} notes.")
        else:
            print("❌ No notes were saved. Check logs for errors.")
            
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        traceback.print_exc() 