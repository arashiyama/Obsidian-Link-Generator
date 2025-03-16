# Obsidian Auto-Linking Tools

A collection of Python scripts for enhancing Obsidian notes with automatic linking, tagging, and relationship discovery.

## Overview

This repository contains a suite of tools designed to enhance the organization and connectedness of notes in an Obsidian vault. Each script has a specific purpose and can be used independently or in combination with others:

1. **semantic_linker.py**: Creates links based on semantic similarity of content
2. **tag_linker.py**: Creates links based on shared tags
3. **genai_linker.py**: Creates links with explanations using Generative AI reasoning
4. **auto_tag_notes.py**: Automatically generates relevant tags for notes

## Requirements

- Python 3.6+
- OpenAI API key (set in .env file)
- Obsidian vault path (set in .env file)

## Setup

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install requirements: `pip install -r requirements.txt`
5. Create a `.env` file with the following content:
   ```
   OPENAI_API_KEY=your_api_key_here
   OBSIDIAN_VAULT_PATH=/path/to/your/obsidian/vault
   ```

## Scripts

### semantic_linker.py

#### Description

This script analyzes all notes in an Obsidian vault, computes embeddings for each note, and creates "Related Notes" sections based on semantic similarity between notes.

#### Features

- Skips processing the "venv" directory during traversal
- Implements caching to avoid regenerating embeddings for unchanged notes
- Uses rate limiting with exponential backoff to handle API limits gracefully
- Parallelizes processing using ThreadPoolExecutor for improved performance
- Chunks large documents into smaller pieces to fit within token limits
- Uses caching at both the chunk and document level for efficiency

#### Example Usage

```bash
python semantic_linker.py
```

### tag_linker.py

#### Description

This script analyzes all notes in an Obsidian vault, extracts tags from each note, and creates "Related Notes (by Tag)" sections based on the number of shared tags.

#### Features

- Extracts tags from both the `#tags:` section and inline tags
- Builds relationships between notes based on shared tags
- Configurable minimum number of shared tags required for relation
- Limits the number of related notes to prevent overwhelming
- Sorts related notes by the number of shared tags

#### Example Usage

```bash
python tag_linker.py
```

### genai_linker.py

#### Description

This script analyzes notes in an Obsidian vault and uses the OpenAI GPT-4 API to identify relevant relationships between notes with detailed explanations. It creates "Related Notes (GenAI)" sections that include relevance scores and explanations of the relationships.

#### Features

- **Smarter Relevance Detection**: Utilizes GPT-4 Turbo to analyze note content and identify related notes
- **Insightful Explanations**: Provides relevance scores (1-10) and detailed explanations for each relationship
- **Efficiency Features**:
  - Caching to avoid redundant API calls
  - Chunking to process notes in small batches
  - Sampling for large vaults
  - Content truncation to fit within API limits

#### Example Usage

```bash
python genai_linker.py
```

### auto_tag_notes.py

#### Description

This script reads each note in an Obsidian vault, uses the OpenAI API to generate appropriate tags based on the content, and updates or adds a `#tags` section to each note.

#### Features

- Skips "venv" directories during processing
- Preserves existing tags when generating new ones
- Adds detailed debugging information to track progress
- Handles API errors gracefully with retries

#### Example Usage

```bash
python auto_tag_notes.py
```

## Comparison of Linking Approaches

### Semantic Linker
- **Pros**: Fast and efficient, uses vector embeddings to find similar content
- **Cons**: No explanations about why notes are related, only identifies surface-level similarity

### Tag Linker
- **Pros**: Very fast, uses explicit tags created by user or auto-tag tool
- **Cons**: Limited to tag overlap, no explanations about relationships

### GenAI Linker
- **Pros**: 
  - Provides detailed reasoning about why notes are related
  - Identifies conceptual and thematic relationships beyond word/tag matching
  - Includes relevance scores to indicate strength of relationships
- **Cons**: 
  - More resource-intensive (API costs, processing time)
  - Processes fewer notes at a time due to API rate limits

## Tips for Usage

For best results, consider using these tools in sequence:

1. Run `auto_tag_notes.py` first to generate relevant tags for all notes
2. Run `tag_linker.py` to create initial relationships based on shared tags
3. Run `semantic_linker.py` to add relationships based on content similarity
4. Run `genai_linker.py` selectively on important notes to add insightful explanations

## Author

Jonathan Care <jonc@lacunae.org> 