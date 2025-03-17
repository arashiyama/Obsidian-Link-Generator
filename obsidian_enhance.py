#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obsidian_enhance.py - A unified tool for enhancing Obsidian notes with auto-tagging and linking

This script combines the functionality of all the individual tools:
- auto_tag_notes.py: Automatically generates tags for notes
- tag_linker.py: Creates links based on shared tags
- semantic_linker.py: Creates links based on semantic similarity
- genai_linker.py: Creates links with explanations using GenAI
- note_categorizer.py: Categorizes notes with visual color tags for graph view

Features:
- Run all tools in optimal sequence or select specific ones to run
- Tracks which notes have been processed by each tool for session persistence
- Provides unified configuration through command-line arguments
- Gracefully handles CTRL+C interrupts for safe termination

Author: Jonathan Care <jonc@lacunae.org>
"""

import os
import sys
import argparse
import json
import time
import random
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import shutil
import utils
import signal_handler

# Import functionality from individual scripts
import auto_tag_notes
import tag_linker
import semantic_linker
import genai_linker
import note_categorizer

# File to track which notes have been processed by each tool
TRACKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tracking")
GENAI_TRACKING_FILE = os.path.join(TRACKING_DIR, "genai_processed_notes.json")
AUTO_TAG_TRACKING_FILE = os.path.join(TRACKING_DIR, "auto_tag_processed_notes.json")
TAG_LINK_TRACKING_FILE = os.path.join(TRACKING_DIR, "tag_link_processed_notes.json")
SEMANTIC_LINK_TRACKING_FILE = os.path.join(TRACKING_DIR, "semantic_link_processed_notes.json")
CATEGORIZER_TRACKING_FILE = os.path.join(TRACKING_DIR, "categorizer_processed_notes.json")

# Ensure tracking directory exists
if not os.path.exists(TRACKING_DIR):
    os.makedirs(TRACKING_DIR)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhance Obsidian vault with auto-tagging and linking")
    
    parser.add_argument("--auto-tag", action="store_true", help="Run auto-tagging on notes")
    parser.add_argument("--tag-link", action="store_true", help="Run tag-based linking")
    parser.add_argument("--semantic-link", action="store_true", help="Run semantic linking")
    parser.add_argument("--genai-link", action="store_true", help="Run GenAI linking")
    parser.add_argument("--categorize", action="store_true", help="Run note categorization for graph coloring")
    parser.add_argument("--all", action="store_true", help="Run all enhancement tools")
    parser.add_argument("--clean", action="store_true", help="Remove all auto-generated links from notes")
    parser.add_argument("--clean-tracking", action="store_true", help="Also clear tracking data when cleaning")
    parser.add_argument("--genai-notes", type=int, default=100, 
                       help="Number of notes to process with GenAI linker (default: 100)")
    parser.add_argument("--force-all", action="store_true", 
                       help="Force processing all notes even if previously processed")
    parser.add_argument("--vault-path", type=str, 
                       help="Path to Obsidian vault (defaults to OBSIDIAN_VAULT_PATH env variable)")
    parser.add_argument("--deduplicate", action="store_true",
                       help="Run deduplication of links and tags across all notes")
    
    return parser.parse_args()

def load_tracking_data(tracking_file):
    """Load tracking data of which notes have been processed by a tool."""
    if os.path.exists(tracking_file):
        try:
            with open(tracking_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading tracking file {tracking_file}: {str(e)}")
            return {"processed_notes": [], "timestamps": []}
    else:
        return {"processed_notes": [], "timestamps": []}

def save_tracking_data(tracking_data, tracking_file):
    """Save tracking data of processed notes."""
    try:
        with open(tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
    except Exception as e:
        print(f"Error writing tracking file {tracking_file}: {str(e)}")

def filter_notes_for_processing(notes, tracking_data, force_all=False):
    """
    Filter notes to process based on tracking data and content changes.
    
    Returns a dictionary mapping path -> boolean indicating whether each note should be processed.
    """
    processed_data = {path: {"processed": False, "changed": False} for path in notes.keys()}
    
    if force_all:
        # Process all notes if forced
        for path in notes.keys():
            processed_data[path]["processed"] = True
            processed_data[path]["changed"] = True
        return processed_data
    
    # Check if we have hashes of previously processed notes
    note_hashes = tracking_data.get("note_hashes", {})
    
    for path, content in notes.items():
        # Skip if it's a directory path
        if os.path.isdir(path):
            continue
            
        current_hash = utils.generate_note_hash(content if isinstance(content, str) else content.get("content", ""))
        
        # Note was not previously processed
        if path not in tracking_data["processed_notes"]:
            processed_data[path]["processed"] = True
            processed_data[path]["changed"] = True
        # Note content has changed since last processing
        elif path in note_hashes and note_hashes[path] != current_hash:
            processed_data[path]["processed"] = True
            processed_data[path]["changed"] = True
            
        # Update the hash
        note_hashes[path] = current_hash
    
    # Update the hashes in tracking data
    tracking_data["note_hashes"] = note_hashes
    
    return processed_data

def select_genai_notes(notes, tracking_data, num_notes=100, force_all=False):
    """Select notes for GenAI processing, prioritizing unprocessed or changed notes."""
    if force_all:
        # If forcing all notes, randomly select from all notes
        all_paths = list(notes.keys())
        random.shuffle(all_paths)
        return all_paths[:num_notes]
    
    processed_data = filter_notes_for_processing(notes, tracking_data)
    
    # Get unprocessed or changed notes
    unprocessed_notes = [path for path, data in processed_data.items() 
                         if data["processed"] and data["changed"]]
    
    print(f"Found {len(unprocessed_notes)} unprocessed or changed notes out of {len(notes)} total")
    
    if len(unprocessed_notes) >= num_notes:
        # If we have enough unprocessed notes, shuffle and select from them
        random.shuffle(unprocessed_notes)
        selected_notes = unprocessed_notes[:num_notes]
    else:
        # If we don't have enough unprocessed notes, use all unprocessed + some processed
        selected_notes = unprocessed_notes.copy()
        
        # Use already processed notes to fill the gap
        processed_notes = [path for path in notes.keys() if path not in unprocessed_notes]
        random.shuffle(processed_notes)
        selected_notes.extend(processed_notes[:num_notes - len(unprocessed_notes)])
    
    return selected_notes

def run_auto_tagging(vault_path, force_all=False):
    """Run auto-tagging with tracking."""
    tracking_data = load_tracking_data(AUTO_TAG_TRACKING_FILE)
    
    # Add timestamp to tracking data
    if "timestamps" not in tracking_data:
        tracking_data["timestamps"] = []
    
    tracking_data["timestamps"].append({
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "operation": "auto_tag"
    })
    
    # Load notes
    notes = auto_tag_notes.load_notes(vault_path)
    if not notes:
        print("No notes found for auto-tagging!")
        return 0
    
    # Filter notes to process
    processed_data = filter_notes_for_processing(notes, tracking_data, force_all)
    notes_to_process = {path: content for path, content in notes.items() 
                       if processed_data[path]["processed"]}
    
    if not notes_to_process:
        print("No new or changed notes to auto-tag. Use --force-all to process all notes.")
        return 0
    
    print(f"Auto-tagging {len(notes_to_process)} out of {len(notes)} total notes...")
    
    # Process the filtered notes
    updated = auto_tag_notes.insert_tags(notes_to_process)
    saved = auto_tag_notes.save_notes(notes_to_process, vault_path)
    
    # Update tracking data
    for path in notes_to_process.keys():
        if path not in tracking_data["processed_notes"]:
            tracking_data["processed_notes"].append(path)
    
    save_tracking_data(tracking_data, AUTO_TAG_TRACKING_FILE)
    
    print(f"Auto-tagging: Processed {saved} notes")
    return saved

def run_tag_linking(vault_path, force_all=False):
    """Run tag-based linking with tracking."""
    tracking_data = load_tracking_data(TAG_LINK_TRACKING_FILE)
    
    # Add timestamp to tracking data
    if "timestamps" not in tracking_data:
        tracking_data["timestamps"] = []
    
    tracking_data["timestamps"].append({
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "operation": "tag_link"
    })
    
    # Load notes
    notes = tag_linker.load_notes(vault_path)
    if not notes:
        print("No notes found for tag linking!")
        return 0
    
    # Extract existing links from all notes
    existing_links = {}
    for path, note in notes.items():
        content = note["content"] if isinstance(note, dict) else note
        existing_links[path] = utils.extract_existing_links(content)
    
    # Extract tags from all notes (this needs to be done on all notes)
    note_tags, tag_to_notes = tag_linker.extract_tags(notes)
    
    # Build relations for all notes (this also needs all notes)
    relations = tag_linker.build_relations(notes, note_tags, tag_to_notes)
    
    # If not forcing, only update notes that have changed or new ones
    if not force_all:
        processed_data = filter_notes_for_processing(notes, tracking_data)
        notes_to_update = {path: note for path, note in notes.items() 
                          if processed_data[path]["processed"]}
        
        # Only keep relations for notes that need updating
        filtered_relations = {path: related for path, related in relations.items() 
                             if path in notes_to_update}
        
        if not filtered_relations:
            print("No new or changed notes to update with tag links. Use --force-all to process all notes.")
            return 0
            
        print(f"Tag linking: Updating {len(filtered_relations)} out of {len(notes)} total notes...")
        updated = tag_linker.update_notes_with_relations(notes, filtered_relations, existing_links)
    else:
        # Process all notes
        print(f"Tag linking: Processing all {len(notes)} notes...")
        updated = tag_linker.update_notes_with_relations(notes, relations, existing_links)
    
    saved = tag_linker.save_notes(notes)
    
    # Update tracking data for all saved notes
    for path in notes.keys():
        if path not in tracking_data["processed_notes"]:
            tracking_data["processed_notes"].append(path)
    
    save_tracking_data(tracking_data, TAG_LINK_TRACKING_FILE)
    
    print(f"Tag linking: Processed {saved} notes")
    return saved

def run_semantic_linking(vault_path, force_all=False):
    """Run semantic linking with tracking."""
    tracking_data = load_tracking_data(SEMANTIC_LINK_TRACKING_FILE)
    
    # Add timestamp to tracking data
    if "timestamps" not in tracking_data:
        tracking_data["timestamps"] = []
    
    tracking_data["timestamps"].append({
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "operation": "semantic_link"
    })
    
    # Load notes
    notes = semantic_linker.load_notes(vault_path)
    if not notes:
        print("No notes found for semantic linking!")
        return 0
    
    # Extract existing links from all notes
    existing_links = {}
    for path, content in notes.items():
        existing_links[path] = utils.extract_existing_links(content)
    
    # For semantic linking, we need to process all notes to compute similarities
    # but we can be selective about which ones we update
    filenames, contents = list(notes.keys()), list(notes.values())
    
    # Get embeddings for all notes (this is necessary for the similarity calculation)
    print("Generating embeddings for all notes (this is necessary for accurate similarity)...")
    embeddings = semantic_linker.get_embeddings(contents)
    
    # If not forcing, only update notes that have changed
    if not force_all:
        processed_data = filter_notes_for_processing(notes, tracking_data)
        notes_to_update = {filename: True for filename, data in processed_data.items() 
                          if data["processed"]}
        
        if not notes_to_update:
            print("No new or changed notes to update with semantic links. Use --force-all to process all notes.")
            return 0
            
        print(f"Semantic linking: Updating {len(notes_to_update)} out of {len(notes)} total notes...")
        
        # Process only a subset of notes but with awareness of all notes' similarities
        subset_links = {}
        for idx, file in enumerate(filenames):
            if file in notes_to_update:
                subset_links[file] = True
                
        semantic_linker.generate_links(notes, embeddings, filenames, existing_links, subset_links)
    else:
        # Process all notes, but still avoid duplicating links
        print(f"Semantic linking: Processing all {len(notes)} notes...")
        semantic_linker.generate_links(notes, embeddings, filenames, existing_links)
    
    # Save the updated notes
    semantic_linker.save_notes(notes, vault_path)
    
    # Update tracking data
    for path in notes.keys():
        if path not in tracking_data["processed_notes"]:
            tracking_data["processed_notes"].append(path)
    
    save_tracking_data(tracking_data, SEMANTIC_LINK_TRACKING_FILE)
    
    print(f"Semantic linking: Processed {len(notes)} notes")
    return len(notes)

def run_custom_genai_linking(notes, tracking_data, num_notes=100, force_all=False):
    """Run genai_linker.py with tracking of processed notes."""
    # Extract summaries for all notes
    summaries = genai_linker.extract_titles_and_summaries(notes)
    
    # Extract existing links from all notes
    existing_links = {}
    for path, note_data in notes.items():
        content = note_data["content"] if isinstance(note_data, dict) else note_data
        existing_links[path] = utils.extract_existing_links(content)
    
    # Select notes to process
    paths_to_process = select_genai_notes(notes, tracking_data, num_notes, force_all)
    
    print(f"Processing {len(paths_to_process)} notes with GenAI linking...")
    updated = 0
    skipped = 0
    
    for path in paths_to_process:
        try:
            # Get existing links for this note
            current_links = existing_links.get(path, [])
            
            # Find relevant notes
            relevant_notes = genai_linker.find_relevant_notes(path, notes, summaries)
            
            if not relevant_notes:
                continue
            
            content = notes[path]["content"]
            
            # Extract existing GenAI related notes section if it exists
            section_text, _ = utils.extract_section(content, "## Related Notes (GenAI)")
            existing_link_entries = []
            if section_text:
                existing_link_entries = section_text.split("\n")
            
            # Create new link entries for relevant notes
            new_link_entries = []
            for rel_note in relevant_notes:
                related_path = rel_note["path"]
                related_filename = notes[related_path]["filename"]
                note_name = os.path.splitext(related_filename)[0]
                
                # Skip if already linked (either in existing GenAI section or anywhere in the note)
                if note_name in current_links:
                    continue
                
                # Format the link with relevance score and reason
                link_entry = f"- [[{note_name}]] (Score: {rel_note['score']}/10)\n  - {rel_note['reason']}"
                new_link_entries.append(link_entry)
                
                # Add to current links to avoid duplicates in future iterations
                current_links.append(note_name)
            
            # If we have no entries to add, skip
            if not (existing_link_entries or new_link_entries):
                continue
            
            # Merge existing and new link entries, avoiding duplicates
            format_func = lambda note_name: f"- [[{note_name}]]"  # Basic formatter if needed
            all_link_entries = utils.merge_links(existing_link_entries, new_link_entries)
            
            # Update the note content with the merged links
            updated_content = utils.replace_section(
                content, 
                "## Related Notes (GenAI)", 
                "\n".join(all_link_entries)
            )
            
            # Update the note content
            notes[path]["content"] = updated_content
            updated += 1
            
            # Add to tracking data with the current content hash
            if path not in tracking_data["processed_notes"]:
                tracking_data["processed_notes"].append(path)
            
            # Update note hash in tracking data
            if "note_hashes" not in tracking_data:
                tracking_data["note_hashes"] = {}
            tracking_data["note_hashes"][path] = utils.generate_note_hash(updated_content)
            
        except Exception as e:
            import traceback
            print(f"Error updating {path}: {str(e)}")
            traceback.print_exc()
            skipped += 1
    
    print(f"GenAI linking: Updated {updated} notes, skipped {skipped} due to errors")
    
    # Save the updated notes
    saved = genai_linker.save_notes(notes)
    
    # Save tracking data
    save_tracking_data(tracking_data, GENAI_TRACKING_FILE)
    
    return updated

def run_note_categorization(vault_path, force_all=False):
    """Run note categorization with tracking."""
    tracking_data = load_tracking_data(CATEGORIZER_TRACKING_FILE)
    
    # Add timestamp to tracking data
    if "timestamps" not in tracking_data:
        tracking_data["timestamps"] = []
    
    tracking_data["timestamps"].append({
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "operation": "categorize"
    })
    
    # Load notes
    notes = note_categorizer.load_notes(vault_path)
    if not notes:
        print("No notes found for categorization!")
        return 0
    
    # Filter notes to process
    if not force_all:
        processed_data = filter_notes_for_processing(notes, tracking_data)
        notes_to_process = {path: content for path, content in notes.items() 
                           if processed_data[path]["processed"]}
        
        if not notes_to_process:
            print("No new or changed notes to categorize. Use --force-all to process all notes.")
            return 0
            
        print(f"Categorizing {len(notes_to_process)} out of {len(notes)} total notes...")
        
        # Replace the notes dict with only the ones we want to process
        # Need to make a copy to keep the original references for when we run custom categorization
        for path in list(notes.keys()):
            if path not in notes_to_process:
                del notes[path]
    
    # Run categorization on the filtered notes
    categorized = note_categorizer.categorize_notes(notes)
    saved = note_categorizer.save_notes(notes, vault_path)
    
    # Update tracking data
    for path in notes.keys():
        if path not in tracking_data["processed_notes"]:
            tracking_data["processed_notes"].append(path)
        
        # Update note hash in tracking data
        if "note_hashes" not in tracking_data:
            tracking_data["note_hashes"] = {}
        tracking_data["note_hashes"][path] = utils.generate_note_hash(notes[path])
    
    save_tracking_data(tracking_data, CATEGORIZER_TRACKING_FILE)
    
    print(f"Note categorization: Processed {saved} notes")
    return saved

def clean_notes(vault_path, clear_tracking=False):
    """
    Remove all auto-generated links from notes.
    Optionally clear tracking data.
    """
    print(f"Cleaning notes in vault: {vault_path}")
    
    # Load all notes
    notes = {}
    count = 0
    skipped = 0
    cleaned = 0
    
    print("Loading notes...")
    for root, dirs, files in os.walk(vault_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
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
    
    # Define patterns for each type of auto-generated links section
    section_headers = [
        "## Related Notes",
        "## Related Notes (by Tag)",
        "## Related Notes (GenAI)"
    ]
    
    # Remove auto-generated links from each note
    for path, content in notes.items():
        original_content = content
        
        # Remove each section if it exists
        for header in section_headers:
            section_content = utils.extract_section(content, header)
            if section_content[0]:  # If section exists
                content = content.replace(section_content[1], "")  # Remove entire section
        
        # Only save if content has changed
        if content != original_content:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                    cleaned += 1
            except Exception as e:
                print(f"Error writing to file {path}: {str(e)}")
    
    print(f"Cleaned {cleaned} notes")
    
    # Clear tracking data if requested
    if clear_tracking:
        if os.path.exists(TRACKING_DIR):
            try:
                shutil.rmtree(TRACKING_DIR)
                os.makedirs(TRACKING_DIR)  # Recreate empty directory
                print("Tracking data cleared")
            except Exception as e:
                print(f"Error clearing tracking data: {str(e)}")
    
    return cleaned

def deduplicate_links_and_tags(vault_path):
    """
    Run dedicated deduplication of links and tags across all notes.
    This removes all duplicate links and tags without modifying the content otherwise.
    """
    print(f"Deduplicating links and tags in vault: {vault_path}")
    
    # Load all notes
    notes = {}
    count = 0
    skipped = 0
    deduplicated = 0
    
    print("Loading notes...")
    for root, dirs, files in os.walk(vault_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
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
    
    # Section headers to check for link deduplication
    link_section_headers = [
        "## Related Notes",
        "## Related Notes (by Tag)",
        "## Related Notes (GenAI)"
    ]
    
    # Process each note
    for path, content in notes.items():
        original_content = content
        modified = False
        
        # Deduplicate tags if the note has a tags section
        if "#tags:" in content.lower():
            # Extract existing tags
            existing_tags = utils.extract_existing_tags(content)
            
            # Deduplicate tags
            unique_tags = utils.deduplicate_tags(existing_tags)
            
            # If deduplication changed the tags, update the content
            if len(unique_tags) != len(existing_tags):
                tags_text = " ".join(unique_tags)
                content = re.sub(r"#tags:.*?(\n\n|\n$|$)", f"#tags: {tags_text}\n", content, flags=re.DOTALL)
                modified = True
        
        # Deduplicate links in each section
        for header in link_section_headers:
            section_text, full_match = utils.extract_section(content, header)
            if section_text:
                # Get link entries from the section
                link_entries = [line for line in section_text.split("\n") if line.strip()]
                
                # Merge links to deduplicate
                deduplicated_links = utils.merge_links(link_entries, [])
                
                # If deduplication changed the links, update the section
                if len(deduplicated_links) != len(link_entries):
                    updated_content = utils.replace_section(content, header, "\n".join(deduplicated_links))
                    content = updated_content
                    modified = True
        
        # Save if changes were made
        if modified:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                    deduplicated += 1
            except Exception as e:
                print(f"Error writing to file {path}: {str(e)}")
    
    print(f"Deduplicated links and tags in {deduplicated} notes")
    return deduplicated

def cleanup_before_exit():
    """Clean up resources before exiting."""
    print("Performing cleanup before exit...")
    # Add any necessary cleanup here, like closing file handles
    # or ensuring tracking data is saved, etc.
    
    print("Cleanup completed. Goodbye!")

def main():
    # Set up clean interrupt handling
    signal_handler.setup_interrupt_handling()
    
    # Register cleanup function to run on exit (whether normal or interrupted)
    signal_handler.register_cleanup_function(cleanup_before_exit)
    
    start_time = time.time()
    print(f"Starting Obsidian vault enhancement at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    args = parse_arguments()
    
    # Set vault path from command line or environment variable
    vault_path = args.vault_path or os.getenv("OBSIDIAN_VAULT_PATH")
    if not vault_path:
        print("No vault path provided. Set OBSIDIAN_VAULT_PATH environment variable or use --vault-path")
        sys.exit(1)
    
    # If no specific tool is selected and not cleaning, do nothing unless --all is set
    if not (args.auto_tag or args.tag_link or args.semantic_link or args.genai_link or 
            args.categorize or args.clean or args.deduplicate) and not args.all:
        print("No tools selected to run. Use --help to see available options.")
        sys.exit(1)
    
    # Run clean if requested
    if args.clean:
        print("\n===== Cleaning Auto-Generated Links =====")
        cleaned = clean_notes(vault_path, args.clean_tracking)
        print(f"Removed auto-generated links from {cleaned} notes")
        if args.clean_tracking:
            print("Tracking data cleared")
        
        # If only cleaning was requested, exit
        if not (args.auto_tag or args.tag_link or args.semantic_link or args.genai_link or 
                args.categorize or args.deduplicate or args.all):
            elapsed_time = time.time() - start_time
            print(f"\nCleaning completed in {elapsed_time:.2f} seconds")
            return
    
    # Run deduplication if requested
    if args.deduplicate:
        print("\n===== Deduplicating Links and Tags =====")
        deduplicated = deduplicate_links_and_tags(vault_path)
        print(f"Deduplicated links and tags in {deduplicated} notes")
        
        # If only deduplication was requested, exit
        if not (args.auto_tag or args.tag_link or args.semantic_link or args.genai_link or 
                args.categorize or args.all):
            elapsed_time = time.time() - start_time
            print(f"\nDeduplication completed in {elapsed_time:.2f} seconds")
            return
    
    # Run categorization
    if args.all or args.categorize:
        print("\n===== Running Note Categorization =====")
        saved = run_note_categorization(vault_path, args.force_all)
        if saved > 0:
            print(f"Note categorization: Processed {saved} notes")
            note_categorizer.print_obsidian_setup_instructions()
    
    # Run auto-tagging
    if args.all or args.auto_tag:
        print("\n===== Running Auto-Tagging =====")
        saved = run_auto_tagging(vault_path, args.force_all)
        if saved > 0:
            print(f"Auto-tagging: Processed {saved} notes")
    
    # Run tag linking
    if args.all or args.tag_link:
        print("\n===== Running Tag Linking =====")
        saved = run_tag_linking(vault_path, args.force_all)
        if saved > 0:
            print(f"Tag linking: Processed {saved} notes")
    
    # Run semantic linking
    if args.all or args.semantic_link:
        print("\n===== Running Semantic Linking =====")
        saved = run_semantic_linking(vault_path, args.force_all)
        if saved > 0:
            print(f"Semantic linking: Processed {saved} notes")
    
    # Run GenAI linking with tracking
    if args.all or args.genai_link:
        print("\n===== Running GenAI Linking =====")
        notes = genai_linker.load_notes(vault_path)
        tracking_data = load_tracking_data(GENAI_TRACKING_FILE)
        
        # Add timestamp to tracking data
        if "timestamps" not in tracking_data:
            tracking_data["timestamps"] = []
        
        tracking_data["timestamps"].append({
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "notes_processed": args.genai_notes
        })
        
        updated = run_custom_genai_linking(notes, tracking_data, args.genai_notes, args.force_all)
        if updated > 0:
            print(f"GenAI linking: Processed {updated} notes")
        
        # Provide coverage statistics
        all_notes = len(notes)
        processed = len(tracking_data["processed_notes"])
        coverage = (processed / all_notes) * 100 if all_notes > 0 else 0
        print(f"Current GenAI coverage: {processed}/{all_notes} notes ({coverage:.1f}%)")
    
    elapsed_time = time.time() - start_time
    print(f"\nEnhancement completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main() 