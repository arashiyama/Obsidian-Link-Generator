#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
storage.py - Persistent storage for Auto Link Obsidian

This module handles loading and saving data to disk, including:
- Note tracking data
- Cache files
- Other persistent state
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime
from pathlib import Path

from .config import config


class Storage:
    """
    Persistent storage manager for Auto Link Obsidian.
    """
    
    def __init__(self):
        """Initialize the storage manager."""
        # Ensure tracking and cache directories exist
        os.makedirs(config["tracking_dir_path"], exist_ok=True)
        os.makedirs(config["cache_dir_path"], exist_ok=True)
    
    def load_tracking_data(self, linker_type: str) -> Dict[str, Any]:
        """
        Load tracking data for a specific linker type.
        
        Args:
            linker_type: Type of linker (e.g., "semantic", "tag", "genai")
            
        Returns:
            Dictionary containing tracking data
        """
        tracking_file = self._get_tracking_file(linker_type)
        
        if os.path.exists(tracking_file):
            try:
                with open(tracking_file, 'r') as f:
                    data = json.load(f)
                
                if config["verbose"]:
                    print(f"Loaded tracking data from {tracking_file}")
                    print(f"Previously processed notes: {len(data.get('processed_notes', []))}")
                
                return data
            except Exception as e:
                print(f"Error reading tracking file {tracking_file}: {str(e)}")
        
        # Return empty tracking data if file doesn't exist or error occurred
        return {"processed_notes": [], "timestamps": [], "note_hashes": {}}
    
    def save_tracking_data(self, linker_type: str, data: Dict[str, Any]) -> bool:
        """
        Save tracking data for a specific linker type.
        
        Args:
            linker_type: Type of linker (e.g., "semantic", "tag", "genai")
            data: Tracking data to save
            
        Returns:
            True if successful, False otherwise
        """
        tracking_file = self._get_tracking_file(linker_type)
        
        try:
            with open(tracking_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            if config["verbose"]:
                print(f"Saved tracking data to {tracking_file}")
                print(f"Total processed notes: {len(data.get('processed_notes', []))}")
            
            return True
        except Exception as e:
            print(f"Error writing tracking file {tracking_file}: {str(e)}")
            return False
    
    def _get_tracking_file(self, linker_type: str) -> str:
        """
        Get the path to the tracking file for a specific linker type.
        
        Args:
            linker_type: Type of linker
            
        Returns:
            Path to the tracking file
        """
        tracking_files = {
            "auto_tag": config["auto_tag_tracking_file"],
            "tag": config["tag_link_tracking_file"],
            "semantic": config["semantic_link_tracking_file"],
            "genai": config["genai_tracking_file"],
            "categorizer": config["categorizer_tracking_file"],
        }
        
        return tracking_files.get(
            linker_type, 
            os.path.join(config["tracking_dir_path"], f"{linker_type}_processed_notes.json")
        )
    
    def update_tracking_timestamp(self, linker_type: str) -> None:
        """
        Add a timestamp to tracking data for a specific linker type.
        
        Args:
            linker_type: Type of linker
        """
        data = self.load_tracking_data(linker_type)
        
        # Add timestamp for this run
        if "timestamps" not in data:
            data["timestamps"] = []
        
        data["timestamps"].append({
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "operation": linker_type
        })
        
        self.save_tracking_data(linker_type, data)
    
    def clear_tracking_data(self, linker_type: Optional[str] = None) -> None:
        """
        Clear tracking data for a specific linker type or all linkers.
        
        Args:
            linker_type: Type of linker, or None to clear all
        """
        if linker_type:
            # Clear for a specific linker type
            tracking_file = self._get_tracking_file(linker_type)
            if os.path.exists(tracking_file):
                try:
                    os.remove(tracking_file)
                    print(f"Cleared tracking data for {linker_type}")
                except Exception as e:
                    print(f"Error clearing tracking data for {linker_type}: {str(e)}")
        else:
            # Clear all tracking data
            for file in os.listdir(config["tracking_dir_path"]):
                if file.endswith("_processed_notes.json"):
                    try:
                        os.remove(os.path.join(config["tracking_dir_path"], file))
                    except Exception as e:
                        print(f"Error clearing tracking data {file}: {str(e)}")
            
            print(f"Cleared all tracking data")
    
    def load_json_cache(self, cache_name: str) -> Dict[str, Any]:
        """
        Load a JSON cache file.
        
        Args:
            cache_name: Name of the cache file (without .json extension)
            
        Returns:
            Dictionary containing cached data
        """
        cache_file = os.path.join(config["cache_dir_path"], f"{cache_name}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                if config["verbose"]:
                    print(f"Loaded cache from {cache_file}")
                
                return data
            except Exception as e:
                print(f"Error reading cache file {cache_file}: {str(e)}")
        
        # Return empty cache if file doesn't exist or error occurred
        return {}
    
    def save_json_cache(self, cache_name: str, data: Dict[str, Any]) -> bool:
        """
        Save data to a JSON cache file.
        
        Args:
            cache_name: Name of the cache file (without .json extension)
            data: Data to save
            
        Returns:
            True if successful, False otherwise
        """
        cache_file = os.path.join(config["cache_dir_path"], f"{cache_name}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            if config["verbose"]:
                print(f"Saved cache to {cache_file}")
            
            return True
        except Exception as e:
            print(f"Error writing cache file {cache_file}: {str(e)}")
            return False
    
    def clear_cache(self, cache_name: Optional[str] = None) -> None:
        """
        Clear a specific cache file or all cache files.
        
        Args:
            cache_name: Name of the cache to clear, or None to clear all
        """
        if cache_name:
            # Clear a specific cache
            cache_file = os.path.join(config["cache_dir_path"], f"{cache_name}.json")
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                    print(f"Cleared cache: {cache_name}")
                except Exception as e:
                    print(f"Error clearing cache {cache_name}: {str(e)}")
        else:
            # Clear all caches
            for file in os.listdir(config["cache_dir_path"]):
                if file.endswith(".json"):
                    try:
                        os.remove(os.path.join(config["cache_dir_path"], file))
                    except Exception as e:
                        print(f"Error clearing cache {file}: {str(e)}")
            
            print(f"Cleared all caches")


# Global storage instance
storage = Storage()