#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto Link Obsidian - A tool for enhancing Obsidian notes with auto-tagging and linking

This package provides functionality for:
- Auto-tagging notes based on content
- Linking notes based on shared tags
- Linking notes based on semantic similarity
- Generating links with explanations using AI
- Categorizing notes for visual graph organization
"""

__version__ = "0.1.0"

# Import core modules
from .core.config import config
from .core.storage import storage
from .core.note import Note
from .core.embedding import get_embedding_provider

# Import linkers
from .linkers.base_linker import linker_registry
from . import linkers