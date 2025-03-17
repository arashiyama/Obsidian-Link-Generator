#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tests.py - Test runner for Auto Link Obsidian

This script discovers and runs all tests in the tests directory.
"""

import unittest
import sys
import os

if __name__ == '__main__':
    # Add current directory to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    print("Running Auto Link Obsidian tests...")
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return non-zero exit code if tests failed
    if not result.wasSuccessful():
        sys.exit(1)
    
    print("All tests passed!")
