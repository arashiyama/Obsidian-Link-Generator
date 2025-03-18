#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - Entry point for Auto Link Obsidian

This script is the main entry point for the Auto Link Obsidian tool.
It orchestrates the different components based on the user's configuration.
"""

import sys
import os
import time
import signal
from datetime import datetime

from auto_link_obsidian import config, linker_registry


def signal_handler(sig, frame):
    """Handle interrupt signals."""
    print("\nInterrupted by user. Cleaning up before exit...")
    sys.exit(0)


def clean_notes(vault_path):
    """
    Remove all auto-generated links from notes.
    
    Args:
        vault_path: Path to the Obsidian vault
        
    Returns:
        Number of notes cleaned
    """
    print(f"Cleaning auto-generated links in {vault_path}")
    
    # For each linker type, get the section header to remove
    section_headers = {
        "semantic": "## Related Notes",
        "tag": "## Related Notes (by Tag)",
        "genai": "## Related Notes (GenAI)",
    }
    
    # TODO: Implement a cleaner that removes sections from notes
    # Should use Note.load_notes() and then remove sections and save
    
    print("Cleaning functionality not yet implemented in the new architecture")
    return 0


def main():
    """
    Main entry point for the Auto Link Obsidian tool.
    
    Parses command line arguments and runs the requested linkers.
    """
    # Register signal handler for clean exit on Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"Starting Auto Link Obsidian at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    
    # Check for valid vault path
    vault_path = config["vault_path"]
    if not vault_path:
        print("No vault path provided. Set OBSIDIAN_VAULT_PATH environment variable or use --vault-path")
        sys.exit(1)
    
    print(f"Using vault path: {vault_path}")
    
    # Run the clean command if requested
    if config.get("clean", False):
        print("\n===== Cleaning Auto-Generated Links =====")
        cleaned = clean_notes(vault_path)
        print(f"Cleaned {cleaned} notes")
        
        # Clear tracking data if requested
        if config.get("clean_tracking", False):
            tracking_dir = config["tracking_dir_path"]
            if os.path.exists(tracking_dir):
                for file in os.listdir(tracking_dir):
                    if file.endswith(".json"):
                        os.remove(os.path.join(tracking_dir, file))
            print("Tracking data cleared")
    
    # Run the requested linkers
    processed_any = False
    
    # 1. Auto-tagging
    if config.get("auto_tag", False):
        print("\n===== Running Auto-Tagging =====")
        if "auto_tag" in linker_registry:
            try:
                auto_tagger = linker_registry["auto_tag"](vault_path)
                saved = auto_tagger.run()
                if saved > 0:
                    print(f"Auto-tagging: Processed {saved} notes")
                processed_any = True
            except Exception as e:
                print(f"Error running auto-tagger: {str(e)}")
        else:
            print("Auto-tagging not yet implemented in the new architecture")
    
    # 2. Tag linking
    if config.get("tag_link", False):
        print("\n===== Running Tag Linking =====")
        if "tag" in linker_registry:
            try:
                tag_linker = linker_registry["tag"](vault_path)
                saved = tag_linker.run()
                if saved > 0:
                    print(f"Tag linking: Processed {saved} notes")
                processed_any = True
            except Exception as e:
                print(f"Error running tag linker: {str(e)}")
        else:
            print("Tag linking not yet implemented in the new architecture")
    
    # 3. Semantic linking
    if config.get("semantic_link", False):
        print("\n===== Running Semantic Linking =====")
        if "semantic" in linker_registry:
            try:
                semantic_linker = linker_registry["semantic"](vault_path)
                saved = semantic_linker.run()
                if saved > 0:
                    print(f"Semantic linking: Processed {saved} notes")
                processed_any = True
            except Exception as e:
                print(f"Error running semantic linker: {str(e)}")
        else:
            print("Semantic linking not available in the registry")
    
    # 4. GenAI linking
    if config.get("genai_link", False):
        print("\n===== Running GenAI Linking =====")
        if "genai" in linker_registry:
            try:
                genai_linker = linker_registry["genai"](vault_path)
                saved = genai_linker.run()
                if saved > 0:
                    print(f"GenAI linking: Processed {saved} notes")
                processed_any = True
            except Exception as e:
                print(f"Error running GenAI linker: {str(e)}")
        else:
            print("GenAI linking not yet implemented in the new architecture")
    
    # 5. Note categorization
    if config.get("categorize", False):
        print("\n===== Running Note Categorization =====")
        if "categorizer" in linker_registry:
            try:
                categorizer = linker_registry["categorizer"](vault_path)
                saved = categorizer.run()
                if saved > 0:
                    print(f"Note categorization: Processed {saved} notes")
                processed_any = True
            except Exception as e:
                print(f"Error running note categorizer: {str(e)}")
        else:
            print("Note categorization not yet implemented in the new architecture")
    
    # If cleaning was not requested and no linkers were run
    if not config.get("clean", False) and not processed_any:
        print("No tools selected to run. Use --help to see available options.")
        print("Available options:")
        print("  --auto-tag         Run auto-tagging on notes")
        print("  --tag-link         Run tag-based linking")
        print("  --semantic-link    Run semantic linking")
        print("  --genai-link       Run GenAI linking")
        print("  --categorize       Run note categorization")
        print("  --all              Run all enhancement tools")
        print("  --clean            Remove all auto-generated links")
        sys.exit(1)
    
    # Print completion message
    elapsed_time = time.time() - start_time
    print(f"\nEnhancement completed in {elapsed_time:.2f} seconds")
    return 0


if __name__ == "__main__":
    sys.exit(main())