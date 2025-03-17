#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
signal_handler.py - Utility for gracefully handling keyboard interrupts

This module provides a signal handler for CTRL+C interrupts that ensures
the program exits cleanly with a proper message instead of showing
a traceback, which can be confusing for users.

Author: Jonathan Care <jonc@lacunae.org>
"""

import os
import sys
import signal
import time

# Flag to track if we're in the process of handling an interrupt
_interrupt_in_progress = False

def _handle_interrupt(signum, frame):
    """
    Signal handler for keyboard interrupts (CTRL+C).
    
    Args:
        signum: Signal number
        frame: Current stack frame
    """
    global _interrupt_in_progress
    
    # Avoid handling the signal twice
    if _interrupt_in_progress:
        # If the user is really insistent by pressing CTRL+C multiple times,
        # exit immediately without cleanup
        print("\n\n! Received multiple interrupts. Exiting immediately.")
        sys.exit(130)  # Standard exit code for SIGINT
    
    _interrupt_in_progress = True
    print("\n\n! Operation interrupted by user. Cleaning up and exiting...")
    
    # Wait a brief moment to allow any current I/O operations to complete
    time.sleep(0.5)
    
    # Exit with the appropriate exit code for SIGINT
    sys.exit(130)

def setup_interrupt_handling():
    """
    Set up the handler for CTRL+C interrupts.
    Call this at the beginning of the main program.
    """
    signal.signal(signal.SIGINT, _handle_interrupt)
    
    # Also handle SIGTERM (sent by some process managers)
    signal.signal(signal.SIGTERM, _handle_interrupt)
    
    print("Signal handlers registered. Press CTRL+C at any time to stop safely.")

def register_cleanup_function(cleanup_func):
    """
    Register a cleanup function to be called before exiting.
    
    Args:
        cleanup_func: Function to be called for cleanup before exit
    """
    # Python's atexit module provides a cleaner way to handle exit cleanup
    # than trying to implement it in the signal handler
    import atexit
    atexit.register(cleanup_func)
    
    return cleanup_func
