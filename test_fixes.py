#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script to verify our fixes to semantic_linker.py and genai_linker.py
"""

import os
import sys
import semantic_linker
import genai_linker

def test_semantic_linker():
    """Test the semantic_linker.py fixes with a small dataset."""
    print("\n===== Testing semantic_linker.py =====")
    
    # Create a small test dataset in different formats
    test_notes = {
        "note1.md": {
            "content": "# Test Note 1\nThis is a test note about artificial intelligence and machine learning.",
            "content_for_embedding": "This is a test note about artificial intelligence and machine learning.",
            "metadata": {"tags": ["AI", "ML"]}
        },
        "note2.md": {
            "content": "# Test Note 2\nThis note discusses Python programming and data structures.",
            "content_for_embedding": "This note discusses Python programming and data structures.",
            "metadata": {"tags": ["Python", "Programming"]}
        },
        "note3.md": "# Test Note 3\nObsidian is a great note-taking tool with many plugins available."
    }
    
    print("Created test notes in different formats")
    print("Generating embeddings...")
    
    # Test the get_embeddings function with mixed format input
    embeddings = semantic_linker.get_embeddings(test_notes)
    
    if embeddings is None:
        print("Error: Failed to generate embeddings")
        return False
    
    print(f"Successfully generated embeddings with shape: {embeddings.shape}")
    
    # Test cosine similarity matrix calculation
    print("Calculating similarity matrix...")
    similarity_matrix = semantic_linker.cosine_similarity_matrix(embeddings)
    
    if similarity_matrix is None:
        print("Error: Failed to calculate similarity matrix")
        return False
    
    print(f"Successfully calculated similarity matrix with shape: {similarity_matrix.shape}")
    
    # Test another scenario - passing list of values instead of dictionary
    print("\nTesting with list of values...")
    test_contents = list(test_notes.values())
    
    embeddings2 = semantic_linker.get_embeddings(test_contents)
    if embeddings2 is None:
        print("Error: Failed to generate embeddings from list input")
        return False
        
    print(f"Successfully generated embeddings from list input with shape: {embeddings2.shape}")
    
    print("Semantic linker tests passed successfully!")
    return True

def test_genai_linker():
    """Test the genai_linker.py fixes with a small dataset."""
    print("\n===== Testing genai_linker.py =====")
    
    # Create a small test dataset in different formats
    test_notes = {
        "note1.md": {
            "filename": "note1.md",
            "content": "# Test Note 1\nThis is a test note about artificial intelligence and machine learning."
        },
        "note2.md": {
            "filename": "note2.md", 
            "content": "# Test Note 2\nThis note discusses Python programming and data structures."
        },
        "note3.md": "# Test Note 3\nObsidian is a great note-taking tool with many plugins available."
    }
    
    print("Created test notes in different formats")
    print("Extracting titles and summaries...")
    
    # Test the extract_titles_and_summaries function
    summaries = genai_linker.extract_titles_and_summaries(test_notes)
    
    if len(summaries) != len(test_notes):
        print(f"Error: Expected {len(test_notes)} summaries, got {len(summaries)}")
        return False
    
    print(f"Successfully extracted titles and summaries for {len(summaries)} notes")
    
    # Check if titles were extracted correctly
    for path, summary in summaries.items():
        print(f"Note: {path}, Title: {summary['title']}")
    
    print("GenAI linker tests passed successfully!")
    return True

def main():
    """Run all tests."""
    successes = 0
    failures = 0
    
    # Test semantic_linker.py
    try:
        if test_semantic_linker():
            successes += 1
        else:
            failures += 1
    except Exception as e:
        print(f"Error testing semantic_linker.py: {str(e)}")
        failures += 1
    
    # Test genai_linker.py
    try:
        if test_genai_linker():
            successes += 1
        else:
            failures += 1
    except Exception as e:
        print(f"Error testing genai_linker.py: {str(e)}")
        failures += 1
    
    print(f"\nTest summary: {successes} passed, {failures} failed")
    
    if failures == 0:
        print("\nAll tests passed! The fixes appear to be working correctly.")
        return 0
    else:
        print("\nSome tests failed. Please check the output for details.")
        return 1

if __name__ == "__main__":
    # Redirect stdout to a file for easier review
    original_stdout = sys.stdout
    with open('test_results.txt', 'w') as f:
        sys.stdout = f
        result = main()
        # Reset stdout
        sys.stdout = original_stdout
        
    # Print a message to the console
    print("Test completed! Results saved to test_results.txt")
    sys.exit(result)
