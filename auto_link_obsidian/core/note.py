#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
note.py - Core Note data model for Auto Link Obsidian

This module defines the Note class, which provides a unified interface
for working with Obsidian notes, including parsing frontmatter, managing links,
and serializing/deserializing note content.
"""

from typing import Dict, List, Optional, Set, Any, Union, TypedDict
import os
import re
import yaml
import hashlib


class Frontmatter(TypedDict, total=False):
    """TypedDict for note frontmatter metadata"""
    tags: List[str]
    category: str
    categories: List[str]
    type: str
    topic: str
    topics: List[str]
    project: str
    area: str


class Note:
    """
    Unified class representing an Obsidian markdown note.
    
    Attributes:
        path: Absolute path to the note file
        filename: Basename of the note file
        title: Note title (extracted from frontmatter or first heading)
        content: Full note content including frontmatter
        body: Note content without frontmatter
        frontmatter: Dict containing parsed YAML frontmatter
        links: Set of wikilinks found in the note
        tags: Set of tags found in the note
    """
    
    def __init__(self, path: str, content: Optional[str] = None):
        """
        Initialize a Note object.
        
        Args:
            path: Absolute path to the note file
            content: Note content (will be loaded from path if not provided)
        """
        self.path = path
        self.filename = os.path.basename(path)
        self.title = os.path.splitext(self.filename)[0]
        
        # Load content if not provided
        if content is None:
            self._load_from_file()
        else:
            self.content = content
            
        # Parse frontmatter and extract body
        self.frontmatter, self.body = self._parse_frontmatter()
        
        # Extract links and tags
        self.links = self._extract_links()
        self.tags = self._extract_tags()
        
        # Generate hash for content tracking
        self._content_hash = self._generate_hash()
        
    def _load_from_file(self) -> None:
        """Load note content from file."""
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                self.content = f.read()
        except Exception as e:
            print(f"Error reading note file {self.path}: {str(e)}")
            self.content = ""
            
    def _parse_frontmatter(self) -> tuple[Frontmatter, str]:
        """
        Parse YAML frontmatter from note content.
        
        Returns:
            tuple: (frontmatter_dict, content_without_frontmatter)
        """
        # Check if content starts with YAML frontmatter (---)
        if self.content.startswith("---"):
            # Find the end of the frontmatter
            end_pos = self.content.find("---", 3)
            if end_pos != -1:
                frontmatter_text = self.content[3:end_pos].strip()
                body = self.content[end_pos+3:].strip()
                
                try:
                    # Parse the YAML frontmatter
                    frontmatter = yaml.safe_load(frontmatter_text)
                    if not isinstance(frontmatter, dict):
                        frontmatter = {}
                    
                    # Extract title from frontmatter if available
                    if "title" in frontmatter:
                        self.title = frontmatter["title"]
                        
                    return frontmatter, body
                except Exception as e:
                    print(f"Error parsing frontmatter for {self.path}: {str(e)}")
        
        # No frontmatter or error parsing
        return {}, self.content
    
    def _extract_links(self) -> Set[str]:
        """
        Extract wiki links from note content.
        
        Returns:
            Set of link targets (without brackets and aliases)
        """
        # Regular expression for Obsidian wiki links [[Target]] or [[Target|Alias]]
        link_pattern = r'\[\[(.*?)(?:\|.*?)?\]\]'
        matches = re.findall(link_pattern, self.content)
        return set(matches)
    
    def _extract_tags(self) -> Set[str]:
        """
        Extract tags from note content.
        
        Returns:
            Set of tags (including # prefix)
        """
        # First check for YAML frontmatter tags
        frontmatter_tags = set()
        if "tags" in self.frontmatter:
            tags = self.frontmatter["tags"]
            if isinstance(tags, list):
                frontmatter_tags = {f"#{tag}" for tag in tags}
            elif isinstance(tags, str):
                # Handle space-separated tags
                frontmatter_tags = {f"#{tag.strip()}" for tag in tags.split() if tag.strip()}
        
        # Extract inline tags with regex
        tag_pattern = r'(?<!\S)#([a-zA-Z0-9_-]+)'
        inline_matches = re.findall(tag_pattern, self.body)
        inline_tags = {f"#{match}" for match in inline_matches}
        
        # Combine all tags
        return frontmatter_tags.union(inline_tags)
    
    def _generate_hash(self) -> str:
        """
        Generate a hash for the note content.
        
        Returns:
            MD5 hash of the content
        """
        return hashlib.md5(self.content.encode('utf-8')).hexdigest()
    
    def get_section(self, header: str) -> Optional[str]:
        """
        Extract a section from the note by its header.
        
        Args:
            header: Section header (e.g., "## Related Notes")
            
        Returns:
            Section content or None if section not found
        """
        if header not in self.body:
            return None
        
        # Find the start of the section (after the header)
        start_pos = self.body.find(header) + len(header)
        
        # Find the start of the next section (if any)
        next_section_match = re.search(r'^#{1,6}\s+\w+', self.body[start_pos:], re.MULTILINE)
        if next_section_match:
            end_pos = start_pos + next_section_match.start()
            section_content = self.body[start_pos:end_pos].strip()
        else:
            # No next section found, take everything until the end
            section_content = self.body[start_pos:].strip()
            
        return section_content
    
    def replace_section(self, header: str, content: str) -> None:
        """
        Replace or add a section in the note.
        
        Args:
            header: Section header (e.g., "## Related Notes")
            content: New section content
        """
        if header in self.body:
            # Find the start of the section
            start_pos = self.body.find(header)
            
            # Find the start of the next section (if any)
            section_end = len(self.body)
            next_section_match = re.search(r'^#{1,6}\s+\w+', self.body[start_pos + len(header):], re.MULTILINE)
            
            if next_section_match:
                section_end = start_pos + len(header) + next_section_match.start()
                
            # Replace the section content
            new_body = (
                self.body[:start_pos + len(header)] + 
                "\n\n" + content + "\n\n" + 
                self.body[section_end:]
            )
            
            # Update the note body and full content
            self.body = new_body
            self._rebuild_content()
        else:
            # Section doesn't exist, add it at the end
            self.body += f"\n\n{header}\n\n{content}\n"
            self._rebuild_content()
            
    def add_links(self, links: List[str], section_header: str = "## Related Notes") -> None:
        """
        Add links to the note in a specific section.
        
        Args:
            links: List of link entries to add
            section_header: The section header where links should be added
        """
        # Get existing section content if it exists
        existing_section = self.get_section(section_header)
        
        existing_links = []
        if existing_section:
            existing_links = [line.strip() for line in existing_section.split("\n") if line.strip()]
        
        # Merge existing and new links, avoiding duplicates
        all_links = existing_links.copy()
        
        # Simple deduplication based on the target note name
        existing_targets = set()
        for link in existing_links:
            # Extract target from markdown link format
            match = re.search(r'\[\[(.*?)(?:\|.*?)?\]\]', link)
            if match:
                existing_targets.add(match.group(1))
        
        # Only add new links that don't already exist
        for link in links:
            match = re.search(r'\[\[(.*?)(?:\|.*?)?\]\]', link)
            if match and match.group(1) not in existing_targets:
                all_links.append(link)
                existing_targets.add(match.group(1))
        
        # Replace the section with merged links
        self.replace_section(section_header, "\n".join(all_links))
            
    def _rebuild_content(self) -> None:
        """
        Rebuild the full note content from frontmatter and body.
        
        Updates the content attribute and regenerates the hash.
        """
        if self.frontmatter:
            # Convert frontmatter dict back to YAML
            frontmatter_yaml = yaml.dump(self.frontmatter, default_flow_style=False)
            self.content = f"---\n{frontmatter_yaml}---\n\n{self.body}"
        else:
            self.content = self.body
            
        # Update the hash
        self._content_hash = self._generate_hash()
        
    def has_changed(self) -> bool:
        """
        Check if the note content has changed since initial loading.
        
        Returns:
            True if content has changed, False otherwise
        """
        current_hash = self._generate_hash()
        return current_hash != self._content_hash
    
    def save(self) -> bool:
        """
        Save the note content back to the file.
        
        Returns:
            True if save was successful, False otherwise
        """
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                f.write(self.content)
            self._content_hash = self._generate_hash()  # Update hash after save
            return True
        except Exception as e:
            print(f"Error saving note {self.path}: {str(e)}")
            return False
            
    def get_summary(self) -> str:
        """
        Generate a brief summary of the note content.
        
        Returns:
            A summary of the note (first paragraph or first ~200 chars)
        """
        # Remove markdown formatting
        text = re.sub(r'#{1,6}\s+', '', self.body)  # Remove headers
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Remove links
        text = re.sub(r'\[\[([^\]|]+)\|?([^\]]*)\]\]', r'\1', text)  # Remove wiki links
        
        # Get first paragraph
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            if len(paragraphs[0]) > 200:
                return paragraphs[0][:200] + "..."
            return paragraphs[0]
        
        # Fallback to first few characters
        if len(text) > 200:
            return text[:200] + "..."
        return text
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: str) -> 'Note':
        """
        Create a Note object from a dictionary representation.
        
        Args:
            data: Dictionary containing note data
            path: Path to the note file
            
        Returns:
            A new Note object
        """
        if isinstance(data, dict) and "content" in data:
            return cls(path, data["content"])
        elif isinstance(data, str):
            return cls(path, data)
        else:
            raise ValueError(f"Invalid data format for note at {path}")