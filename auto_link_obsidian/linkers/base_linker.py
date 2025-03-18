#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_linker.py - Abstract base class for all linker implementations

This module defines the BaseLiner class, which provides the common interface
and shared functionality for all types of note linking strategies.
"""

import os
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set, Type
from datetime import datetime

from ..core.note import Note
from ..core.config import config


class BaseLinker(ABC):
    """
    Abstract base class for all note linkers.
    
    All linking strategies (semantic, tag-based, GenAI) should inherit from this
    class and implement the required abstract methods.
    """
    
    # Type of the linker - subclasses should override
    TYPE = "base"
    
    # Name of the section to add links to
    SECTION_HEADER = "## Related Notes"
    
    def __init__(self, vault_path: Optional[str] = None):
        """
        Initialize the linker.
        
        Args:
            vault_path: Path to the Obsidian vault (defaults to config value)
        """
        self.vault_path = vault_path or config["vault_path"]
        if not self.vault_path:
            raise ValueError("No vault path provided")
        
        # Dictionary to store loaded notes
        self.notes: Dict[str, Note] = {}
        
        # Tracking data for processed notes
        self.tracking_data = self._load_tracking_data()
    
    @property
    def tracking_file(self) -> str:
        """Get the path to the tracking file for this linker type."""
        tracking_files = {
            "auto_tag": config["auto_tag_tracking_file"],
            "tag": config["tag_link_tracking_file"],
            "semantic": config["semantic_link_tracking_file"],
            "genai": config["genai_tracking_file"],
            "categorizer": config["categorizer_tracking_file"],
        }
        return tracking_files.get(self.TYPE, os.path.join(config["tracking_dir_path"], f"{self.TYPE}_processed_notes.json"))
    
    def run(self) -> int:
        """
        Run the linker on all notes.
        
        This method orchestrates the linking process:
        1. Load notes from the vault
        2. Process notes (implemented by subclasses)
        3. Save notes back to disk
        4. Update tracking data
        
        Returns:
            The number of notes processed
        """
        # Start timing
        start_time = time.time()
        
        # Load notes
        self.load_notes()
        if not self.notes:
            print(f"No notes found in {self.vault_path}")
            return 0
        
        # Process notes
        processed_notes = self.process_notes()
        
        # Save notes back to disk
        saved_count = self.save_notes()
        
        # Update tracking data
        self._update_tracking_data()
        
        # Print statistics
        elapsed_time = time.time() - start_time
        print(f"{self.TYPE.capitalize()} linking completed in {elapsed_time:.2f}s - Processed {saved_count} notes")
        
        return saved_count
    
    def load_notes(self) -> Dict[str, Note]:
        """
        Load all markdown notes from the vault.
        
        Returns:
            Dictionary mapping file paths to Note objects
        """
        print(f"Loading notes from {self.vault_path}")
        
        # Walk through all directories and files in the vault
        for root, dirs, files in os.walk(self.vault_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith(".md"):
                    try:
                        path = os.path.join(root, file)
                        # Create a Note object
                        note = Note(path)
                        self.notes[path] = note
                    except Exception as e:
                        print(f"Error reading {file}: {str(e)}")
        
        print(f"Loaded {len(self.notes)} notes from vault")
        return self.notes
    
    @abstractmethod
    def process_notes(self) -> int:
        """
        Process notes to generate links.
        
        This is the main method where each linker implementation should
        implement its specific linking strategy.
        
        Returns:
            Number of notes processed
        """
        pass
    
    def filter_notes_to_process(self) -> Dict[str, Note]:
        """
        Filter notes to determine which ones need processing.
        
        Uses tracking data to skip notes that haven't changed since
        last processing, unless force_all is enabled.
        
        Returns:
            Dictionary of notes that need processing
        """
        if config["force_all"]:
            print("Force-all mode enabled, processing all notes")
            return self.notes
        
        # Dictionary to store notes to process
        notes_to_process: Dict[str, Note] = {}
        
        # Get previously processed notes
        processed_notes = set(self.tracking_data.get("processed_notes", []))
        note_hashes = self.tracking_data.get("note_hashes", {})
        
        new_count = 0
        changed_count = 0
        
        # Check each note
        for path, note in self.notes.items():
            # Never processed before
            if path not in processed_notes:
                notes_to_process[path] = note
                new_count += 1
                continue
                
            # Check if content has changed
            current_hash = note._content_hash
            if path in note_hashes and note_hashes[path] != current_hash:
                notes_to_process[path] = note
                changed_count += 1
                continue
                
        print(f"Filtering: {new_count} new notes, {changed_count} changed notes, {len(self.notes) - new_count - changed_count} unchanged")
        return notes_to_process
    
    def save_notes(self) -> int:
        """
        Save all notes back to disk.
        
        Returns:
            Number of notes successfully saved
        """
        print(f"Saving notes to {self.vault_path}")
        
        saved_count = 0
        error_count = 0
        
        for path, note in self.notes.items():
            if note.has_changed():  # Only save if the note has changed
                if note.save():
                    saved_count += 1
                else:
                    error_count += 1
        
        print(f"Saved {saved_count} notes ({error_count} errors)")
        return saved_count
    
    def _load_tracking_data(self) -> Dict[str, Any]:
        """
        Load tracking data from disk.
        
        Returns:
            Dictionary containing tracking data
        """
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    data = json.load(f)
                
                if config["verbose"]:
                    print(f"Loaded tracking data from {self.tracking_file}")
                    print(f"Previously processed notes: {len(data.get('processed_notes', []))}")
                
                return data
            except Exception as e:
                print(f"Error reading tracking file {self.tracking_file}: {str(e)}")
        
        # Return empty tracking data if file doesn't exist or error occurred
        return {"processed_notes": [], "timestamps": [], "note_hashes": {}}
    
    def _update_tracking_data(self) -> None:
        """Update tracking data with newly processed notes."""
        # Add timestamp for this run
        if "timestamps" not in self.tracking_data:
            self.tracking_data["timestamps"] = []
        
        self.tracking_data["timestamps"].append({
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "operation": self.TYPE
        })
        
        # Ensure note_hashes exists
        if "note_hashes" not in self.tracking_data:
            self.tracking_data["note_hashes"] = {}
        
        # Update processed notes and their hashes
        for path, note in self.notes.items():
            if path not in self.tracking_data["processed_notes"]:
                self.tracking_data["processed_notes"].append(path)
            self.tracking_data["note_hashes"][path] = note._content_hash
        
        # Save updated tracking data
        self._save_tracking_data()
    
    def _save_tracking_data(self) -> None:
        """Save tracking data to disk."""
        try:
            with open(self.tracking_file, 'w') as f:
                json.dump(self.tracking_data, f, indent=2)
            
            if config["verbose"]:
                print(f"Saved tracking data to {self.tracking_file}")
                print(f"Total processed notes: {len(self.tracking_data.get('processed_notes', []))}")
        except Exception as e:
            print(f"Error writing tracking file {self.tracking_file}: {str(e)}")


# Registry of linker implementations
linker_registry: Dict[str, Type[BaseLinker]] = {}