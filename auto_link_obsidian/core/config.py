#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py - Configuration management for Auto Link Obsidian

This module centralizes configuration settings loaded from multiple sources:
1. Default values
2. Configuration file
3. Environment variables
4. Command line arguments (overrides all others)
"""

import os
import sys
import argparse
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path


class Config:
    """
    Configuration manager for Auto Link Obsidian.
    
    This class provides a unified interface for all application settings,
    with prioritized loading from multiple sources.
    """
    
    # Default configuration values
    DEFAULTS = {
        # General settings
        "vault_path": "",  # Must be provided via ENV var, config file, or CLI
        "verbose": False,
        
        # Note processing settings
        "force_all": False,
        "batch_size": 50,
        
        # Feature settings
        "auto_tag": False,
        "tag_link": False,
        "semantic_link": False,
        "genai_link": False,
        "categorize": False,
        
        # Semantic linking settings
        "similarity_threshold": 0.75,
        "embedding_batch_size": 20,
        "embedding_model": "text-embedding-3-small",
        
        # GenAI linking settings
        "genai_notes": 100,
        "genai_model": "gpt-3.5-turbo",
        
        # Paths
        "tracking_dir": ".tracking",
        "cache_dir": ".cache",
        
        # Important metadata fields for similarity calculations
        "important_metadata_fields": [
            "tags", "category", "categories", "type", 
            "topic", "topics", "project", "area"
        ],
    }
    
    def __init__(self, load_from_args: bool = False, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            load_from_args: Whether to load configuration from command line arguments
            config_file: Path to a configuration file to load from
        """
        # Start with default configuration
        self._config = self.DEFAULTS.copy()
        
        # Load from configuration file if specified
        if config_file:
            self.load_from_file(config_file)
        else:
            # Try to load from default locations
            default_locations = [
                os.path.join(os.getcwd(), "config.yaml"),
                os.path.expanduser("~/.config/auto_link_obsidian/config.yaml"),
            ]
            for path in default_locations:
                if os.path.exists(path):
                    self.load_from_file(path)
                    break
        
        # Load from environment variables
        self.load_from_env()
        
        # Load from command line arguments if requested
        if load_from_args:
            self.load_from_args()
            
        # Set computed attributes
        self.set_computed_attributes()
    
    def load_from_file(self, config_file: str) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            config_file: Path to the configuration file
        """
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                
            if isinstance(config_data, dict):
                # Update configuration with file values
                for key, value in config_data.items():
                    if key in self._config:
                        self._config[key] = value
                print(f"Loaded configuration from {config_file}")
        except Exception as e:
            print(f"Error loading configuration from {config_file}: {str(e)}")
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Map config keys to environment variable names
        env_mapping = {
            "vault_path": "OBSIDIAN_VAULT_PATH",
            "embedding_model": "OPENAI_EMBEDDING_MODEL",
            "genai_model": "OPENAI_GENAI_MODEL",
            "similarity_threshold": "SIMILARITY_THRESHOLD",
        }
        
        # Load values from environment
        for config_key, env_var in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Convert to appropriate type
                if isinstance(self._config[config_key], bool):
                    value = value.lower() in ('true', 'yes', '1')
                elif isinstance(self._config[config_key], int):
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                elif isinstance(self._config[config_key], float):
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                
                self._config[config_key] = value
    
    def load_from_args(self) -> None:
        """Load configuration from command line arguments."""
        parser = argparse.ArgumentParser(description="Enhance Obsidian vault with auto-tagging and linking")
        
        # General options
        parser.add_argument("--vault-path", type=str, help="Path to Obsidian vault")
        parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
        parser.add_argument("--force-all", action="store_true", help="Force processing all notes regardless of cache")
        parser.add_argument("--batch-size", type=int, help="Maximum notes to process in a batch")
        
        # Feature selection
        parser.add_argument("--auto-tag", action="store_true", help="Run auto-tagging on notes")
        parser.add_argument("--tag-link", action="store_true", help="Run tag-based linking")
        parser.add_argument("--semantic-link", action="store_true", help="Run semantic linking")
        parser.add_argument("--genai-link", action="store_true", help="Run GenAI linking")
        parser.add_argument("--categorize", action="store_true", help="Run note categorization")
        parser.add_argument("--all", action="store_true", help="Run all enhancement tools")
        
        # Semantic linking options
        parser.add_argument("--similarity-threshold", type=float, help="Similarity threshold for semantic linking")
        parser.add_argument("--embedding-model", type=str, help="Model to use for embeddings")
        
        # GenAI linking options
        parser.add_argument("--genai-notes", type=int, help="Number of notes to process with GenAI")
        parser.add_argument("--genai-model", type=str, help="Model to use for GenAI processing")
        
        # Maintenance options
        parser.add_argument("--clean", action="store_true", help="Remove all auto-generated links from notes")
        parser.add_argument("--clean-tracking", action="store_true", help="Clear tracking data when cleaning")
        parser.add_argument("--deduplicate", action="store_true", help="Deduplicate links and tags across notes")
        
        # Parse arguments
        args = parser.parse_args()
        
        # Update configuration with command line values (only for non-None values)
        for key, value in vars(args).items():
            # Convert CLI argument names to config keys (e.g., vault_path instead of vault-path)
            config_key = key.replace('-', '_')
            
            # Special handling for the "all" flag
            if config_key == "all" and value:
                # Enable all features if --all is specified
                self._config["auto_tag"] = True
                self._config["tag_link"] = True
                self._config["semantic_link"] = True
                self._config["genai_link"] = True
                self._config["categorize"] = True
            elif value is not None and config_key in self._config:
                self._config[config_key] = value
    
    def set_computed_attributes(self) -> None:
        """Set attributes that are computed from other configuration values."""
        # Get script directory
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Set tracking directory (within script directory by default)
        tracking_dir = os.path.join(script_dir, self._config["tracking_dir"])
        self._config["tracking_dir_path"] = tracking_dir
        
        # Create tracking directory if it doesn't exist
        if not os.path.exists(tracking_dir):
            os.makedirs(tracking_dir)
            
        # Set tracking file paths
        self._config["auto_tag_tracking_file"] = os.path.join(tracking_dir, "auto_tag_processed_notes.json")
        self._config["tag_link_tracking_file"] = os.path.join(tracking_dir, "tag_link_processed_notes.json")
        self._config["semantic_link_tracking_file"] = os.path.join(tracking_dir, "semantic_link_processed_notes.json")
        self._config["genai_tracking_file"] = os.path.join(tracking_dir, "genai_processed_notes.json")
        self._config["categorizer_tracking_file"] = os.path.join(tracking_dir, "categorizer_processed_notes.json")
        
        # Set cache directory
        cache_dir = os.path.join(script_dir, self._config["cache_dir"])
        self._config["cache_dir_path"] = cache_dir
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Set cache file paths
        self._config["embeddings_cache_file"] = os.path.join(cache_dir, "embeddings_cache.json")
    
    def __getitem__(self, key: str) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            
        Returns:
            The configured value for the key
        """
        return self._config.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: New value
        """
        self._config[key] = value
        
    def __contains__(self, key: str) -> bool:
        """
        Check if a configuration key exists.
        
        Args:
            key: Configuration key to check
            
        Returns:
            True if the key exists, False otherwise
        """
        return key in self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with a default fallback.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            The configured value or default
        """
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get a dictionary representation of the configuration.
        
        Returns:
            Dictionary containing all configuration values
        """
        return self._config.copy()
    
    def save_to_file(self, config_file: str) -> bool:
        """
        Save the current configuration to a file.
        
        Args:
            config_file: Path where to save the configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Filter out computed attributes
            save_config = {k: v for k, v in self._config.items() if not k.endswith("_path") and not k.endswith("_file")}
            
            with open(config_file, 'w') as f:
                yaml.dump(save_config, f, default_flow_style=False)
            return True
        except Exception as e:
            print(f"Error saving configuration to {config_file}: {str(e)}")
            return False


# Global configuration instance
config = Config(load_from_args=True)