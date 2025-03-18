#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linkers package - Link generation modules for Auto Link Obsidian

This package contains the different link generation strategies:
- Semantic linking: links based on content similarity
- Tag linking: links based on shared tags
- GenAI linking: links with AI-generated explanations
- Auto-tagging: automatically generate tags for notes
"""

# Import linkers to register them
from . import semantic
from . import tag
from . import genai
# The following will be implemented later:
# from . import auto_tag

# Import the registry for easy access
from .base_linker import linker_registry, BaseLinker
