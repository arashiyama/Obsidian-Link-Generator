#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tag.py - Tag-based linking implementation for Auto Link Obsidian

This module implements linking between notes based on shared tags.
"""

from typing import Dict, List, Set, Optional, Any
import re
from collections import defaultdict

from ..core.note import Note
from ..core.config import config
from ..utils.markdown import extract_tags
from .base_linker import BaseLinker, linker_registry


class TagLinker(BaseLinker):
    """
    Linker implementation that connects notes based on shared tags.
    """
    
    TYPE = "tag"
    SECTION_HEADER = "## Related Notes (by Tag)"
    
    def __init__(self, vault_path: Optional[str] = None):
        """
        Initialize the tag linker.
        
        Args:
            vault_path: Path to the Obsidian vault
        """
        super().__init__(vault_path)
        
        # Set minimum number of shared tags to create a link
        self.min_shared_tags = config.get("min_shared_tags", 1)
        
        # Print configuration info
        print(f"Initialized tag linker with minimum shared tags: {self.min_shared_tags}")
    
    def process_notes(self) -> int:
        """
        Process notes to generate links based on shared tags.
        
        Returns:
            Number of notes processed
        """
        # Filter notes to determine which ones need processing
        notes_to_process = self.filter_notes_to_process()
        
        if not notes_to_process:
            print("No notes to process")
            return 0
            
        print(f"Processing {len(notes_to_process)} notes out of {len(self.notes)} total")
        
        # Build tag indices for all notes
        print("Building tag indices...")
        note_tags = self._extract_all_tags()
        tag_to_notes = self._build_tag_index(note_tags)
        
        # Build relationships between notes based on tags
        print("Building note relationships based on shared tags...")
        relations = self._build_relations(note_tags, tag_to_notes)
        
        # Update notes with relationships
        print("Updating notes with tag-based links...")
        processed_count = 0
        
        for note_path, related_notes in relations.items():
            # Skip notes that don't need processing
            if note_path not in notes_to_process:
                continue
                
            try:
                # Get the note object
                note = self.notes[note_path]
                
                # Format link entries for related notes
                link_entries = []
                for rel_path, shared in related_notes:
                    # Get the related note
                    related_note = self.notes[rel_path]
                    
                    # Format shared tags
                    tags_text = ", ".join(shared)
                    
                    link_entry = f"- [[{related_note.title}]] (Shared tags: {tags_text})"
                    link_entries.append(link_entry)
                
                # Only update if we have links to add
                if link_entries:
                    # Add links to the note
                    note.add_links(link_entries, self.SECTION_HEADER)
                    processed_count += 1
                    
            except Exception as e:
                print(f"Error processing note {note_path}: {str(e)}")
        
        print(f"Added tag-based links to {processed_count} notes")
        return processed_count
    
    def _extract_all_tags(self) -> Dict[str, Set[str]]:
        """
        Extract tags from all notes.
        
        Returns:
            Dictionary mapping note paths to sets of tags
        """
        note_tags = {}
        for path, note in self.notes.items():
            tags = set()
            
            # Extract tags from frontmatter
            if "tags" in note.frontmatter:
                fm_tags = note.frontmatter["tags"]
                if isinstance(fm_tags, list):
                    tags.update(tag.lower() for tag in fm_tags)
                elif isinstance(fm_tags, str):
                    tags.update(tag.lower() for tag in fm_tags.split())
            
            # Extract inline tags (without the # prefix)
            for tag in note.tags:
                if tag.startswith('#'):
                    tags.add(tag[1:].lower())
                else:
                    tags.add(tag.lower())
            
            note_tags[path] = tags
        
        return note_tags
    
    def _build_tag_index(self, note_tags: Dict[str, Set[str]]) -> Dict[str, List[str]]:
        """
        Build an index mapping tags to notes containing them.
        
        Args:
            note_tags: Dictionary mapping note paths to sets of tags
            
        Returns:
            Dictionary mapping tags to lists of note paths
        """
        tag_to_notes = defaultdict(list)
        
        for path, tags in note_tags.items():
            for tag in tags:
                tag_to_notes[tag].append(path)
        
        return tag_to_notes
    
    def _build_relations(self, note_tags: Dict[str, Set[str]], tag_to_notes: Dict[str, List[str]]) -> Dict[str, List[tuple]]:
        """
        Build relationships between notes based on shared tags.
        
        Args:
            note_tags: Dictionary mapping note paths to sets of tags
            tag_to_notes: Dictionary mapping tags to lists of note paths
            
        Returns:
            Dictionary mapping note paths to lists of (related_path, shared_tags) tuples
        """
        relations = {}
        
        for path, tags in note_tags.items():
            # Dictionary to track related notes and their shared tags
            related = defaultdict(set)
            
            # Find related notes through shared tags
            for tag in tags:
                for related_path in tag_to_notes[tag]:
                    # Skip self-reference
                    if related_path == path:
                        continue
                    
                    # Add the shared tag
                    related[related_path].add(tag)
            
            # Filter by minimum shared tags
            filtered_related = [
                (rel_path, shared_tags) 
                for rel_path, shared_tags in related.items() 
                if len(shared_tags) >= self.min_shared_tags
            ]
            
            # Sort by number of shared tags (descending)
            sorted_related = sorted(filtered_related, key=lambda x: len(x[1]), reverse=True)
            
            # Store the relationships
            relations[path] = sorted_related
        
        return relations


# Register the linker
linker_registry["tag"] = TagLinker