#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_semantic_linker.py - Unit tests for the semantic_linker.py module

This module contains unit tests for the semantic linking functionality.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import tempfile
import json

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the OpenAI module
sys.modules['openai'] = MagicMock()

import semantic_linker

class TestFrontmatterExtraction(unittest.TestCase):
    """Test cases for frontmatter extraction."""
    
    def test_extract_frontmatter_with_valid_yaml(self):
        """Test extracting valid YAML frontmatter."""
        content = """---
title: Test Note
tags:
  - python
  - testing
category: Development
---

# Test Note

This is the actual content.
"""
        frontmatter, content_without_frontmatter = semantic_linker.extract_frontmatter(content)
        
        self.assertEqual(frontmatter["title"], "Test Note")
        self.assertEqual(frontmatter["category"], "Development")
        self.assertIn("python", frontmatter["tags"])
        self.assertIn("testing", frontmatter["tags"])
        self.assertIn("# Test Note", content_without_frontmatter)
        
    def test_extract_frontmatter_without_yaml(self):
        """Test extracting frontmatter when none exists."""
        content = """# Test Note

This is a note without frontmatter.
"""
        frontmatter, content_without_frontmatter = semantic_linker.extract_frontmatter(content)
        
        self.assertEqual(frontmatter, {})
        self.assertEqual(content_without_frontmatter, content)
        
    def test_extract_frontmatter_with_invalid_yaml(self):
        """Test extracting invalid YAML frontmatter."""
        content = """---
title: Test Note
tags: [ not closed properly
---

# Test Note

This is the actual content.
"""
        frontmatter, content_without_frontmatter = semantic_linker.extract_frontmatter(content)
        
        self.assertEqual(frontmatter, {})
        self.assertEqual(content_without_frontmatter, content)

class TestMetadataPreparation(unittest.TestCase):
    """Test cases for metadata preparation."""
    
    def test_prepare_metadata_empty(self):
        """Test preparing empty metadata."""
        metadata = {}
        result = semantic_linker.prepare_metadata_for_embedding(metadata)
        self.assertEqual(result, "")
        
    def test_prepare_metadata_with_fields(self):
        """Test preparing metadata with various field types."""
        metadata = {
            "tags": ["python", "testing", "ml"],
            "category": "Development"
            # Only include fields that are in IMPORTANT_METADATA_FIELDS
        }
        
        result = semantic_linker.prepare_metadata_for_embedding(metadata)
        
        self.assertIn("tags: python, testing, ml", result)
        self.assertIn("category: Development", result)

class TestCosineSimilarityMatrix(unittest.TestCase):
    """Test cases for cosine similarity calculation."""
    
    def test_cosine_similarity_matrix(self):
        """Test calculating cosine similarity matrix."""
        # Create simple test embeddings
        embeddings = np.array([
            [1, 0, 0],  # Vector pointing in x direction
            [0, 1, 0],  # Vector pointing in y direction
            [0, 0, 1],  # Vector pointing in z direction
            [1, 1, 0]   # Vector pointing in xy plane (45 degrees)
        ])
        
        # Calculate similarity matrix
        similarity = semantic_linker.cosine_similarity_matrix(embeddings)
        
        # Verify dimensions
        self.assertEqual(similarity.shape, (4, 4))
        
        # Verify diagonal is 1 (self-similarity)
        for i in range(4):
            self.assertAlmostEqual(similarity[i, i], 1.0)
        
        # Verify orthogonal vectors have zero similarity
        self.assertAlmostEqual(similarity[0, 1], 0.0)
        self.assertAlmostEqual(similarity[0, 2], 0.0)
        self.assertAlmostEqual(similarity[1, 2], 0.0)
        
        # Verify 45-degree vector has expected similarity
        self.assertAlmostEqual(similarity[0, 3], 1.0 / np.sqrt(2))
        self.assertAlmostEqual(similarity[1, 3], 1.0 / np.sqrt(2))

if __name__ == '__main__':
    unittest.main()
