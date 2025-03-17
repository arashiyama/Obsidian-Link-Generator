# Obsidian Auto-Linking Tools

A collection of Python scripts for enhancing Obsidian notes with automatic linking, tagging, and relationship discovery.

## Overview

This repository contains a suite of tools designed to enhance the organization and connectedness of notes in an Obsidian vault. Each script has a specific purpose and can be used independently or in combination with others:

1. **semantic_linker.py**: Creates links based on semantic similarity of content
2. **tag_linker.py**: Creates links based on shared tags
3. **genai_linker.py**: Creates links with explanations using Generative AI reasoning
4. **auto_tag_notes.py**: Automatically generates relevant tags for notes
5. **note_categorizer.py**: Categorizes notes and adds color tags for Graph View visualization
6. **obsidian_enhance.py**: Unified tool that combines all the above functionalities

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

### note_categorizer.py

#### Description

This script analyzes each note in an Obsidian vault using OpenAI, determines the most appropriate category for the note, and adds special color tags. These tags can be used to visually distinguish different types of notes in Obsidian's graph view.

#### Features

- **AI-Powered Categorization**: Uses OpenAI to intelligently categorize notes into types (person, concept, project, etc.)
- **Visual Graph Enhancements**: Adds special tags that can be used to color-code notes in Obsidian's graph view
- **Multiple Categories**: Supports 10 different note categories:
  - concept
  - person
  - project
  - research
  - reference
  - journal
  - note
  - tool
  - event
  - place
- **Efficiency Features**:
  - Caching to avoid redundant API calls
  - Smart content truncation
  - Provides setup instructions for Obsidian graph view

#### Example Usage

```bash
python note_categorizer.py
```

After running, follow the provided instructions to configure Obsidian's graph view to display different colors for each category.

### obsidian_enhance.py

#### Description

This unified script combines the functionality of all the individual tools into a single command-line interface. It allows you to run any combination of the tools in the optimal sequence and tracks which notes have been processed by each tool for complete vault coverage over time.

#### Features

- Run all tools in the optimal sequence with a single command
- Select specific tools to run using command-line flags
- **Smart Session Memory**:
  - Tracks which notes have been processed by each tool across sessions
  - Only processes new or modified notes by default to save time and API costs
  - Detects when note content has changed and prioritizes those notes for reprocessing
  - Maintains separate tracking data for each tool
- Prioritizes processing unprocessed notes when using the GenAI linker
- Provides progress statistics for GenAI linking coverage
- Customizable number of notes to process with GenAI linker per run
- **Clean Function**: Easily remove all auto-generated links and optionally reset tracking data

#### Example Usage

```bash
# Run all tools with default settings
python obsidian_enhance.py --all

# Run only specific tools
python obsidian_enhance.py --auto-tag --tag-link

# Process more notes with GenAI linker
python obsidian_enhance.py --genai-link --genai-notes 200

# Run only the categorization tool
python obsidian_enhance.py --categorize

# Force processing all notes even if previously processed
python obsidian_enhance.py --all --force-all

# Clean all auto-generated links from notes
python obsidian_enhance.py --clean

# Clean all auto-generated links and reset tracking data
python obsidian_enhance.py --clean --clean-tracking

# Specify a different vault path
python obsidian_enhance.py --all --vault-path /path/to/vault
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

## Visual Organization with Categories

The note categorizer enhances your Obsidian experience by:

1. **Visual Graph Organization**: Use different colors to distinguish note types in Graph View
2. **Improved Navigation**: Quickly identify the type of note from its appearance
3. **Structural Understanding**: See patterns in your knowledge base based on note types
4. **Filter by Category**: Use category tags to filter and find specific types of notes

To set up colors in Graph View:
1. Open Obsidian and navigate to Graph View
2. Click the settings icon
3. In the Groups section, create a group for each category tag
4. Choose distinctive colors for each category

## Session Memory and Incremental Processing

The tools have been designed to intelligently remember which notes have been processed across multiple sessions:

1. **Efficiency**: By default, only new or modified notes are processed in subsequent runs
2. **Change Detection**: The system tracks content hashes to detect when notes have changed
3. **Progress Persistence**: Processing progress is saved between sessions in `.tracking` files
4. **Coverage Tracking**: Statistics show overall coverage for tools like GenAI linker
5. **Forced Processing**: Use the `--force-all` flag to reprocess all notes regardless of history

Benefits of this approach:
- Significant time and API cost savings in large vaults
- Incremental improvement of your knowledge base
- Full coverage of GenAI linking over multiple sessions
- No duplicate work when rerunning the tools

## Tips for Usage

For best results, consider using these tools in sequence:

1. Run `note_categorizer.py` first to create a visual organization system
2. Run `auto_tag_notes.py` to generate relevant tags for all notes
3. Run `tag_linker.py` to create initial relationships based on shared tags
4. Run `semantic_linker.py` to add relationships based on content similarity
5. Run `genai_linker.py` selectively on important notes to add insightful explanations

Alternatively, use the unified `obsidian_enhance.py` script to run all or selected tools in the optimal sequence.

### Cleaning and Starting Fresh

If you want to remove all auto-generated links and start fresh:

1. Use the `--clean` flag to remove all auto-generated links:
   ```bash
   python obsidian_enhance.py --clean
   ```

2. To also reset the tracking data and start completely fresh:
   ```bash
   python obsidian_enhance.py --clean --clean-tracking
   ```
   
3. You can combine cleaning with immediately reprocessing:
   ```bash
   python obsidian_enhance.py --clean --all
   ```

This is useful when you want to remove all automated content and regenerate everything with updated settings or after making significant changes to your vault.

## Contributing and Bug Reports

If you encounter any issues or have suggestions for improvements, please report them through GitHub:

1. Go to the [Issues](https://github.com/jonathancare/obsidian-auto-linking/issues) page
2. Click on "New Issue"
3. Choose the appropriate issue template (Bug Report or Feature Request)
4. Fill in the necessary details

When reporting bugs, please include:
- The specific script that has the issue
- Steps to reproduce the problem
- Expected behavior vs. actual behavior
- Your operating system and Python version
- Any error messages or logs

### Contributing Code

If you'd like to contribute code improvements:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a Pull Request with a clear description of the changes

All contributions are welcome!

## Support

If you find these tools useful, consider buying me a coffee:

<a href="https://buymeacoffee.com/jonathancare" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

## Author

Jonathan Care <jonc@lacunae.org> 