#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
markdown.py - Utilities for working with Obsidian-flavored Markdown

This module provides functions for manipulating Markdown content in ways
specific to Obsidian, such as handling wiki links, sections, and tags.
"""

import re
from typing import Dict, List, Set, Optional, Tuple, Any


def extract_links(content: str) -> Set[str]:
    """
    Extract wiki links from Markdown content.
    
    Args:
        content: Markdown content to extract links from
        
    Returns:
        Set of link targets (without brackets and aliases)
    """
    # Regular expression for Obsidian wiki links [[Target]] or [[Target|Alias]]
    link_pattern = r'\[\[(.*?)(?:\|.*?)?\]\]'
    matches = re.findall(link_pattern, content)
    return set(matches)


def extract_tags(content: str) -> Set[str]:
    """
    Extract tags from Markdown content.
    
    Args:
        content: Markdown content to extract tags from
        
    Returns:
        Set of tags (including # prefix)
    """
    # Extract tags from frontmatter
    frontmatter_tags = set()
    if content.startswith("---"):
        end_pos = content.find("---", 3)
        if end_pos != -1:
            frontmatter = content[3:end_pos].strip()
            # Look for tags: in frontmatter
            tags_match = re.search(r'tags:[ \t]*(.+?)(\n|$)', frontmatter)
            if tags_match:
                tags_text = tags_match.group(1).strip()
                # Handle YAML array format with leading dash
                if re.search(r'^\s*-', tags_text):
                    # YAML array format
                    for line in re.findall(r'-\s*([^\n]+)', tags_text):
                        tag = line.strip().strip("'\"")
                        if tag:
                            frontmatter_tags.add(f"#{tag}")
                else:
                    # Space-separated format
                    for tag in tags_text.split():
                        tag = tag.strip().strip("'\"")
                        if tag:
                            frontmatter_tags.add(f"#{tag}")
    
    # Extract inline tags
    # Using negative lookbehind to avoid matching in URLs or words
    tag_pattern = r'(?<!\S)#([a-zA-Z0-9_-]+)'
    inline_matches = re.findall(tag_pattern, content)
    inline_tags = {f"#{match}" for match in inline_matches}
    
    # Combine all tags
    return frontmatter_tags.union(inline_tags)


def extract_section(content: str, header: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract a section from Markdown content by its header.
    
    Args:
        content: Markdown content to extract from
        header: Section header (e.g., "## Related Notes")
        
    Returns:
        Tuple of (section_content, full_match) or (None, None) if section not found
    """
    if header not in content:
        return None, None
    
    # Find the start of the section
    start_pos = content.find(header)
    
    # Find the end of the header line
    header_end = content.find("\n", start_pos)
    if header_end == -1:
        header_end = len(content)
    
    # Find the start of the next section (if any)
    next_section_match = re.search(r'^#{1,6}\s+\w+', content[header_end:], re.MULTILINE)
    
    if next_section_match:
        section_end = header_end + next_section_match.start()
        full_match = content[start_pos:section_end].strip()
        section_content = content[header_end:section_end].strip()
    else:
        # No next section found, take everything until the end
        full_match = content[start_pos:].strip()
        section_content = content[header_end:].strip()
    
    return section_content, full_match


def replace_section(content: str, header: str, new_content: str) -> str:
    """
    Replace a section in Markdown content or add it if it doesn't exist.
    
    Args:
        content: Markdown content to modify
        header: Section header (e.g., "## Related Notes")
        new_content: New content for the section
        
    Returns:
        Modified Markdown content
    """
    section_content, full_match = extract_section(content, header)
    
    if full_match:
        # Replace the section
        replacement = f"{header}\n\n{new_content}" if new_content else header
        return content.replace(full_match, replacement)
    else:
        # Add the section at the end
        separator = "\n\n" if content and not content.endswith("\n\n") else ""
        section_text = f"{separator}{header}\n\n{new_content}" if new_content else f"{separator}{header}"
        return content + section_text


def merge_links(existing_links: List[str], new_links: List[str]) -> List[str]:
    """
    Merge two lists of link entries, avoiding duplicates based on the link target.
    
    Args:
        existing_links: List of existing link entries
        new_links: List of new link entries to add
        
    Returns:
        Merged list of link entries
    """
    # Track link targets we've already seen
    seen_targets = set()
    merged_links = []
    
    # Process existing links first
    for link in existing_links:
        if not link.strip():
            continue
            
        # Extract the target from the wiki link
        match = re.search(r'\[\[(.*?)(?:\|.*?)?\]\]', link)
        if match:
            target = match.group(1)
            if target not in seen_targets:
                merged_links.append(link)
                seen_targets.add(target)
    
    # Process new links, adding only if target not already seen
    for link in new_links:
        if not link.strip():
            continue
            
        match = re.search(r'\[\[(.*?)(?:\|.*?)?\]\]', link)
        if match:
            target = match.group(1)
            if target not in seen_targets:
                merged_links.append(link)
                seen_targets.add(target)
    
    return merged_links


def deduplicate_tags(tags: List[str]) -> List[str]:
    """
    Deduplicate tags, ignoring case variations.
    
    Args:
        tags: List of tags to deduplicate
        
    Returns:
        Deduplicated list of tags
    """
    unique_tags = {}
    for tag in tags:
        # Remove # prefix for comparison if present
        tag_name = tag[1:] if tag.startswith("#") else tag
        lower_tag = tag_name.lower()
        
        # Keep the original case version
        if lower_tag not in unique_tags:
            unique_tags[lower_tag] = tag
    
    # Convert back to list, preserving original order as much as possible
    result = []
    seen = set()
    
    # First pass: keep tags in original order
    for tag in tags:
        tag_name = tag[1:] if tag.startswith("#") else tag
        lower_tag = tag_name.lower()
        if lower_tag not in seen:
            result.append(unique_tags[lower_tag])
            seen.add(lower_tag)
    
    return result


def get_note_filename(path: str) -> str:
    """
    Extract the note name from a file path.
    
    Args:
        path: Path to the note file
        
    Returns:
        Note name without extension
    """
    # Get the base filename without directory
    filename = path.split("/")[-1]
    
    # Remove extension
    note_name = filename.rsplit(".", 1)[0]
    
    return note_name