#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py - Shared utility functions for Obsidian enhancement tools

This module provides common utility functions used across the various
Obsidian enhancement tools, including link extraction, tag management,
and content manipulation functions.

Author: Jonathan Care <jonc@lacunae.org>
"""

import os
import re
import hashlib

def extract_existing_links(content):
    """
    Extract all existing wiki links from a note.
    
    Args:
        content (str): The note content to extract links from
        
    Returns:
        list: List of link targets (without brackets)
    """
    links = []
    
    # Find all wiki links in the content
    for match in re.finditer(r'\[\[(.*?)(?:\|.*?)?\]\]', content):
        link = match.group(1).strip()
        links.append(link)
    
    return links

def extract_existing_tags(content):
    """
    Extract all existing tags from a note (both from #tags section and inline).
    
    Args:
        content (str): The note content to extract tags from
        
    Returns:
        list: List of tags (with # prefix)
    """
    existing_tags = []
    
    # Extract tags from #tags section if it exists
    tags_section_match = re.search(r'#tags:\s*(.*?)(\n\n|\n$|$)', content, re.IGNORECASE | re.DOTALL)
    if tags_section_match:
        tags_text = tags_section_match.group(1).strip()
        # Extract tags from the tags section
        tags_from_section = [tag.strip() for tag in re.findall(r'#\w+', tags_text)]
        existing_tags.extend(tags_from_section)
    
    # Find other inline tags in the document
    inline_tags = [f"#{tag}" for tag in re.findall(r'#([a-zA-Z0-9_]+)', content)]
    
    # Combine all tags and remove duplicates while preserving order
    all_tags = []
    for tag in existing_tags + inline_tags:
        tag_lower = tag.lower()  # Case-insensitive comparison
        if not any(t.lower() == tag_lower for t in all_tags):
            all_tags.append(tag)
    
    return all_tags

def extract_section(content, section_header):
    """
    Extract a section from note content.
    
    Args:
        content (str): The note content
        section_header (str): The section header to extract (e.g., "## Related Notes")
        
    Returns:
        tuple: (section_text, full_match) or (None, None) if section not found
    """
    # Escape any regex special characters in the section header
    escaped_header = re.escape(section_header)
    
    # Match the section and its content until the next section or end of file
    pattern = f"{escaped_header}\\n(.*?)(?=\\n## |\\n#|\\Z)"
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return match.group(1), match.group(0)
    return None, None

def replace_section(content, section_header, new_section_content):
    """
    Replace a section in note content or add it if it doesn't exist.
    
    Args:
        content (str): The note content
        section_header (str): The section header (e.g., "## Related Notes")
        new_section_content (str): The new content for the section
        
    Returns:
        str: Updated content
    """
    # Check if section exists
    escaped_header = re.escape(section_header)
    if re.search(f"{escaped_header}\\n", content):
        # Replace existing section
        pattern = f"{escaped_header}.*?(?=\\n## |\\n#|\\Z)"
        new_section = f"{section_header}\n{new_section_content}\n\n"
        updated_content = re.sub(pattern, new_section, content, flags=re.DOTALL)
        return updated_content
    else:
        # Add new section at the end
        if content.strip() and not content.endswith("\n\n"):
            if content.endswith("\n"):
                section_text = f"\n{section_header}\n{new_section_content}\n"
            else:
                section_text = f"\n\n{section_header}\n{new_section_content}\n"
        else:
            section_text = f"{section_header}\n{new_section_content}\n"
        return content + section_text

def merge_links(existing_links, new_links, format_func=None):
    """
    Merge existing and new links, avoiding duplicates.
    
    Args:
        existing_links (list): List of existing link entries (full lines with formatting)
        new_links (list): List of new links to add
        format_func (callable, optional): Function to format new link entries
        
    Returns:
        list: Merged list of link entries with duplicates removed
    """
    # Extract note names from existing entries
    note_names_added = set()
    merged_links = []
    
    # Process existing entries first
    for entry in existing_links:
        if not entry.strip():
            continue
            
        link_match = re.search(r'\[\[(.*?)\]\]', entry)
        if link_match:
            note_name = link_match.group(1)
            if note_name.lower() not in (name.lower() for name in note_names_added):
                merged_links.append(entry)
                note_names_added.add(note_name)
    
    # Process new links
    for link in new_links:
        if isinstance(link, str) and "[[" in link:
            # Link is already formatted
            link_match = re.search(r'\[\[(.*?)\]\]', link)
            if link_match:
                note_name = link_match.group(1)
        else:
            # Link is just the note name
            note_name = link
        
        # Skip if already added
        if note_name.lower() in (name.lower() for name in note_names_added):
            continue
            
        # Format the link if needed
        if format_func and not (isinstance(link, str) and link.startswith("-")):
            entry = format_func(link)
        elif isinstance(link, str) and link.startswith("- [["):
            entry = link
        else:
            entry = f"- [[{link}]]"
            
        merged_links.append(entry)
        note_names_added.add(note_name)
    
    return merged_links

def deduplicate_tags(tags):
    """
    Deduplicate a list of tags case-insensitively.
    
    Args:
        tags (list): List of tags (with # prefix)
        
    Returns:
        list: Deduplicated list of tags
    """
    unique_tags = []
    added_tags_lower = set()
    
    for tag in tags:
        # Ensure it has # prefix
        if not tag.startswith('#'):
            tag = f"#{tag}"
            
        tag_lower = tag.lower()
        if tag_lower not in added_tags_lower:
            unique_tags.append(tag)
            added_tags_lower.add(tag_lower)
    
    return unique_tags

def generate_note_hash(note_content):
    """
    Generate a hash for note content to track changes.
    
    Args:
        note_content (str): The note content to hash
        
    Returns:
        str: MD5 hash of the content
    """
    return hashlib.md5(note_content.encode('utf-8')).hexdigest()

def get_note_filename(path):
    """
    Extract the note name without extension from a file path.
    
    Args:
        path (str): The path to the note file
        
    Returns:
        str: The note name without extension
    """
    return os.path.splitext(os.path.basename(path))[0] 