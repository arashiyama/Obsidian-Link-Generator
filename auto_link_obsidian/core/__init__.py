#\!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core package - Core functionality for Auto Link Obsidian

This package contains the core functionality:
- Note: Data model for Obsidian notes
- Config: Configuration management
- Embedding: Embedding provider interface and implementations
- Storage: Persistent storage management
"""

from .note import Note, Frontmatter
from .config import config, Config
from .embedding import EmbeddingProvider, get_embedding_provider
from .storage import storage, Storage
