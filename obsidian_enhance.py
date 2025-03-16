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
- Tracks which notes have been processed by genai_linker for full vault coverage over time
- Provides unified configuration through command-line arguments

Author: Jonathan Care <jonc@lacunae.org>
"""

import os
import sys
import argparse
import json
import time
import random
from datetime import datetime

# Import functionality from individual scripts
import auto_tag_notes
import tag_linker
import semantic_linker
import genai_linker
import note_categorizer

# File to track which notes have been processed by genai_linker
TRACKING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "genai_processed_notes.json")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhance Obsidian vault with auto-tagging and linking")
    
    parser.add_argument("--auto-tag", action="store_true", help="Run auto-tagging on notes")
    parser.add_argument("--tag-link", action="store_true", help="Run tag-based linking")
    parser.add_argument("--semantic-link", action="store_true", help="Run semantic linking")
    parser.add_argument("--genai-link", action="store_true", help="Run GenAI linking")
    parser.add_argument("--categorize", action="store_true", help="Run note categorization for graph coloring")
    parser.add_argument("--all", action="store_true", help="Run all enhancement tools")
    parser.add_argument("--genai-notes", type=int, default=100, 
                       help="Number of notes to process with GenAI linker (default: 100)")
    parser.add_argument("--vault-path", type=str, 
                       help="Path to Obsidian vault (defaults to OBSIDIAN_VAULT_PATH env variable)")
    
    return parser.parse_args()

def load_tracking_data():
    """Load tracking data of which notes have been processed by genai_linker."""
    if os.path.exists(TRACKING_FILE):
        try:
            with open(TRACKING_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading tracking file: {str(e)}")
            return {"processed_notes": []}
    else:
        return {"processed_notes": []}

def save_tracking_data(tracking_data):
    """Save tracking data of processed notes."""
    try:
        with open(TRACKING_FILE, 'w') as f:
            json.dump(tracking_data, f, indent=2)
    except Exception as e:
        print(f"Error writing tracking file: {str(e)}")

def select_genai_notes(notes, tracking_data, num_notes=100):
    """Select notes for GenAI processing, prioritizing unprocessed notes."""
    processed_notes = set(tracking_data["processed_notes"])
    all_paths = list(notes.keys())
    
    # Find unprocessed notes first
    unprocessed_notes = [path for path in all_paths if path not in processed_notes]
    print(f"Found {len(unprocessed_notes)} unprocessed notes out of {len(all_paths)} total")
    
    if len(unprocessed_notes) >= num_notes:
        # If we have enough unprocessed notes, shuffle and select from them
        random.shuffle(unprocessed_notes)
        selected_notes = unprocessed_notes[:num_notes]
    else:
        # If we don't have enough unprocessed notes, use all unprocessed + some processed
        selected_notes = unprocessed_notes.copy()
        
        # Use already processed notes to fill the gap
        processed_to_use = list(processed_notes)
        random.shuffle(processed_to_use)
        selected_notes.extend(processed_to_use[:num_notes - len(unprocessed_notes)])
    
    return selected_notes

def run_custom_genai_linking(notes, tracking_data, num_notes=100):
    """Run genai_linker.py with tracking of processed notes."""
    # Extract summaries for all notes
    summaries = genai_linker.extract_titles_and_summaries(notes)
    
    # Select notes to process
    paths_to_process = select_genai_notes(notes, tracking_data, num_notes)
    
    print(f"Processing {len(paths_to_process)} notes with GenAI linking...")
    updated = 0
    skipped = 0
    
    for path in paths_to_process:
        try:
            # Find relevant notes
            relevant_notes = genai_linker.find_relevant_notes(path, notes, summaries)
            
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
                import re
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
            
            # Add to tracking data
            if path not in tracking_data["processed_notes"]:
                tracking_data["processed_notes"].append(path)
            
        except Exception as e:
            import traceback
            print(f"Error updating {path}: {str(e)}")
            traceback.print_exc()
            skipped += 1
    
    print(f"GenAI linking: Updated {updated} notes, skipped {skipped} due to errors")
    
    # Save the updated notes
    saved = genai_linker.save_notes(notes)
    
    # Save tracking data
    save_tracking_data(tracking_data)
    
    return updated

def main():
    start_time = time.time()
    print(f"Starting Obsidian vault enhancement at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    args = parse_arguments()
    
    # Set vault path from command line or environment variable
    vault_path = args.vault_path or os.getenv("OBSIDIAN_VAULT_PATH")
    if not vault_path:
        print("No vault path provided. Set OBSIDIAN_VAULT_PATH environment variable or use --vault-path")
        sys.exit(1)
    
    # If no specific tool is selected, do nothing unless --all is set
    if not (args.auto_tag or args.tag_link or args.semantic_link or args.genai_link or args.categorize) and not args.all:
        print("No tools selected to run. Use --help to see available options.")
        sys.exit(1)
    
    # Run categorization
    if args.all or args.categorize:
        print("\n===== Running Note Categorization =====")
        notes = note_categorizer.load_notes(vault_path)
        categorized = note_categorizer.categorize_notes(notes)
        saved = note_categorizer.save_notes(notes, vault_path)
        print(f"Note categorization: Processed {saved} notes")
    
    # Run auto-tagging
    if args.all or args.auto_tag:
        print("\n===== Running Auto-Tagging =====")
        notes = auto_tag_notes.load_notes(vault_path)
        updated = auto_tag_notes.insert_tags(notes)
        saved = auto_tag_notes.save_notes(notes, vault_path)
        print(f"Auto-tagging: Processed {saved} notes")
    
    # Run tag linking
    if args.all or args.tag_link:
        print("\n===== Running Tag Linking =====")
        notes = tag_linker.load_notes(vault_path)
        note_tags, tag_to_notes = tag_linker.extract_tags(notes)
        relations = tag_linker.build_relations(notes, note_tags, tag_to_notes)
        updated = tag_linker.update_notes_with_relations(notes, relations)
        saved = tag_linker.save_notes(notes)
        print(f"Tag linking: Processed {saved} notes")
    
    # Run semantic linking
    if args.all or args.semantic_link:
        print("\n===== Running Semantic Linking =====")
        notes = semantic_linker.load_notes(vault_path)
        filenames, contents = list(notes.keys()), list(notes.values())
        embeddings = semantic_linker.get_embeddings(contents)
        semantic_linker.generate_links(notes, embeddings, filenames)
        semantic_linker.save_notes(notes, vault_path)
        print(f"Semantic linking: Processed {len(notes)} notes")
    
    # Run GenAI linking with tracking
    if args.all or args.genai_link:
        print("\n===== Running GenAI Linking =====")
        notes = genai_linker.load_notes(vault_path)
        tracking_data = load_tracking_data()
        
        # Add timestamp to tracking data
        if "timestamps" not in tracking_data:
            tracking_data["timestamps"] = []
        
        tracking_data["timestamps"].append({
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "notes_processed": args.genai_notes
        })
        
        updated = run_custom_genai_linking(notes, tracking_data, args.genai_notes)
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