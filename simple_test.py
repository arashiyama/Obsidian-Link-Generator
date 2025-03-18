#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script to verify our fixes to semantic_linker.py and genai_linker.py
"""

import semantic_linker
import genai_linker

def main():
    print("======= TESTING FIXES =======")
    
    # Test 1: semantic_linker with list input
    print("\nTest 1: semantic_linker.get_embeddings with list input")
    test_notes = [
        "This is a test note about artificial intelligence.",
        "This note discusses Python programming.",
        "Obsidian is a great note-taking tool."
    ]
    print("Input: List of 3 text strings")
    
    # Try get_embeddings with list input
    try:
        embeddings = semantic_linker.get_embeddings(test_notes)
        if embeddings is not None:
            print(f"Success! Got embeddings with shape: {embeddings.shape}")
        else:
            print("Failed: embeddings is None")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
    
    # Test 2: genai_linker with mixed input
    print("\nTest 2: genai_linker.extract_titles_and_summaries with mixed input")
    test_notes = {
        "note1.md": {
            "filename": "note1.md", 
            "content": "# Test Note 1\nContent 1"
        },
        "note2.md": "# Test Note 2\nContent 2" 
    }
    print("Input: Dictionary with mixed formats")
    
    # Try extract_titles_and_summaries with mixed input
    try:
        summaries = genai_linker.extract_titles_and_summaries(test_notes)
        print(f"Success! Got {len(summaries)} summaries")
        for path, summary in summaries.items():
            print(f"  {path}: {summary['title']}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
    
    print("\n======= TEST COMPLETED =======")

if __name__ == "__main__":
    main()
