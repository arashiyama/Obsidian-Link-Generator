#\!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils package - Utility functions for Auto Link Obsidian

This package contains various utility functions and helpers:
- Markdown utilities for manipulating content
- Frontmatter utilities for parsing YAML headers
- General utility functions
"""

from .markdown import (
    extract_links,
    extract_tags,
    extract_section,
    replace_section,
    merge_links,
    deduplicate_tags,
    get_note_filename
)
