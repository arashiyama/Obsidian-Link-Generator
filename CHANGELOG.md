# Changelog

All notable changes to the Auto Link Obsidian project will be documented in this file.

## [Unreleased]

## [0.3.0] - 2025-03-05

### Added
- Link deduplication across all linking tools
- Cross-tool awareness to prevent multiple sections from linking the same note
- Enhanced preservation of existing links while adding new ones
- Created signal_handler.py for graceful handling of CTRL+C interrupts
- Added cleanup functions to all scripts to ensure proper resource release
- User-friendly messages during program interruption
- Protection against data corruption from abrupt termination
- Enhanced note categorization using GenAI with taxonomy system
- Command-line arguments for categorizer script
- `--taxonomy-path` option to specify where to store the category taxonomy
- `--sample-size` option to control how many notes to process per run
- Improved documentation in README.md to reflect all features

### Changed
- Updated all scripts to use the signal handler module
- Improved error handling throughout the codebase
- Enhanced user experience during program interruption

### Fixed
- Fixed issue with duplicate links being added in multiple sections
- Resolved errors in semantic linking when processing notes without sufficient content
- Fixed import error in semantic_linker.py for cosine_similarity function
- Fixed bugs related to processing notes in batches
- Resolved NameError for cosine_similarity function in obsidian_enhance.py

## [0.2.0] - 2025-01-28

### Added
- Enhanced session memory system to efficiently process only new or modified notes
- `--force-all` flag to override session memory and process all notes
- `--only-new` flag to process only new notes, skipping modified ones
- Improved tracking of note changes using content hashing
- Created reusable utils.py module with shared utility functions:
  - extract_existing_links() for extracting wiki links from notes
  - extract_existing_tags() for gathering tags from notes
  - extract_section() and replace_section() for handling note sections
  - merge_links() for combining links without duplicates
  - deduplicate_tags() for removing duplicate tags
  - generate_note_hash() for tracking content changes
- `--deduplicate` flag to run dedicated deduplication of links and tags

### Changed
- Optimized GenAI linking to prioritize unprocessed or modified notes
- Improved tag extraction logic
- Refactored code to use shared utility functions
- Improved modularity and maintainability across all scripts
- DRY (Don't Repeat Yourself) improvements by centralizing common functionality

### Fixed
- Fixed bugs in link extraction and deduplication
- Improved error handling in GenAI API calls
- Minor bug fixes and performance improvements

## [0.1.0] - 2024-12-24

### Added
- Initial release with core functionality
- Auto-tagging tool for generating relevant tags based on note content
- Semantic linking tool for creating links based on content similarity
- Tag linking tool for connecting notes that share common tags
- GenAI linking tool for discovering and explaining connections between notes
- Note categorizer for organizing notes in the graph view
- Unified script (obsidian_enhance.py) to run all enhancement features
- Basic session memory to track processed notes
- Basic tag and link deduplication within individual tools 