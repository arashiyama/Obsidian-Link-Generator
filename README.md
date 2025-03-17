# Auto Link Obsidian

A suite of Python scripts to enhance Obsidian notes with automatic tagging, linking, and categorization.

## Features

- **Auto-tagging**: Generate relevant tags based on note content using GenAI
- **Semantic linking**: Create links between semantically related notes
- **Tag-based linking**: Link notes that share the same tags
- **GenAI linking**: Discover and explain connections between notes using GPT
- **Note categorization**: Categorize notes for better organization
- **Session memory**: Efficiently process only new or modified notes
- **Interrupt handling**: Gracefully handle CTRL+C interrupts with proper cleanup

## Requirements

- Python 3.8+
- OpenAI API key (for GenAI features)
- Obsidian vault

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your configuration (see below)

## Configuration

Create a `.env` file with the following variables:

```
OPENAI_API_KEY=your_api_key_here
OBSIDIAN_VAULT_PATH=/path/to/your/obsidian/vault
```

## Usage

The main script runs all enhancement features:

```bash
python obsidian_enhance.py
```

Individual scripts can be run separately:

```bash
python auto_tag_notes.py
python semantic_linker.py
python tag_linker.py
python genai_linker.py
python note_categorizer.py
```

## Script Descriptions

### obsidian_enhance.py

Main script that orchestrates all enhancement features. Includes session memory to efficiently process only new or modified notes.

Options:
- `--force-all`: Process all notes regardless of modification status
- `--only-new`: Process only new notes, not modified ones
- `--dry-run`: Report what would be done without making changes

### auto_tag_notes.py

Automatically generates tags for notes based on their content using GPT.

### semantic_linker.py

Creates links between semantically related notes based on content similarity.

### tag_linker.py

Links notes that share common tags.

### genai_linker.py

Uses GPT to discover and explain connections between notes, including a relevance score.

### note_categorizer.py

Categorizes notes into a taxonomy of categories for better organization.

Options:
- `--vault-path`: Path to Obsidian vault (if not set in .env)
- `--taxonomy-path`: Path to store category taxonomy (default: category_taxonomy.json)
- `--sample-size`: Number of random notes to process (default: 5)

## Utility Modules

### utils.py

Contains shared utility functions for managing links, tags, and note sections:

- Extracting existing links and tags from notes
- Deduplicating links and tags
- Handling note sections
- Generating note hashes for change detection

### signal_handler.py

Provides graceful handling of CTRL+C interrupts:

- Allows scripts to register cleanup functions
- Ensures resources are properly released
- Provides user-friendly messages on interrupt
- Prevents data corruption from abrupt termination

## Advanced Features

### Session Memory

The scripts use a `.obs_processed` file to track which notes have been processed. This enables:

- Efficient processing of only new or modified notes
- Significant performance improvement for large vaults
- Complete vault coverage over multiple runs

You can override this with the `--force-all` flag to process all notes regardless of their processing history.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License 