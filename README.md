# Auto Link Obsidian

A suite of Python scripts to enhance Obsidian notes with automatic tagging, linking, and categorization.

## Features

- **Auto-tagging**: Generate relevant tags based on note content using GPT
- **Semantic linking**: Create links between semantically related notes using OpenAI embeddings
- **Tag-based linking**: Link notes that share the same tags
- **GenAI linking**: Discover and explain connections between notes using GPT
- **Note categorization**: Categorize notes for better organization
- **Session memory**: Efficiently process only new or modified notes
- **Interrupt handling**: Gracefully handle CTRL+C interrupts with proper cleanup
- **Deduplication**: Prevent duplicate links across different linking methods

## Requirements

- Python 3.8+
- OpenAI API key (for GenAI features and embeddings)
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
- `--auto-tag`: Run auto-tagging on notes
- `--tag-link`: Run tag-based linking
- `--semantic-link`: Run semantic linking
- `--genai-link`: Run GenAI linking
- `--categorize`: Run note categorization
- `--all`: Run all enhancement tools
- `--clean`: Remove all auto-generated links from notes
- `--clean-tracking`: Also clear tracking data when cleaning
- `--deduplicate`: Run deduplication of links and tags across all notes

### auto_tag_notes.py

Automatically generates tags for notes based on their content using GPT.

### semantic_linker.py

Creates links between semantically related notes based on semantic similarity. Now uses OpenAI's embedding API for significantly improved understanding of note relationships, with batch processing and automatic retry logic for API calls.

### tag_linker.py

Links notes that share common tags (default minimum: 2 matching tags).

### genai_linker.py

Uses GPT to discover and explain connections between notes, including a relevance score and natural language explanation of the relationship.

### note_categorizer.py

Categorizes notes into a taxonomy of categories for better organization and graph visualization.

Options:
- `--vault-path`: Path to Obsidian vault (if not set in .env)
- `--taxonomy-path`: Path to store category taxonomy (default: category_taxonomy.json)
- `--sample-size`: Number of random notes to process (default: 5)

The category taxonomy is stored in `category_taxonomy.json`, which is automatically generated during the categorization process. This file is specific to your vault and is excluded from version control (added to `.gitignore`). A sample file `category_taxonomy.example.json` is provided to show the expected structure.

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

### State-of-the-Art Semantic Understanding

The semantic linking feature now uses OpenAI's text-embedding-3-small model to generate high-quality embeddings, resulting in much more accurate and relevant note connections. The system:

- Processes notes in batches to optimize API usage
- Includes automatic retry logic for API calls
- Calculates cosine similarity between all note embeddings
- Sorts and prioritizes the most semantically similar connections
- Limits results to the top 10 most relevant notes to avoid information overload

### Smart Embedding Cache

To improve performance and reduce API costs, the system now implements a smart caching mechanism for embeddings:

- Automatically caches embeddings for each note based on content hash
- Only generates new embeddings for notes that have changed
- Provides cache hit statistics for performance monitoring
- Significantly reduces processing time for subsequent runs
- Minimizes OpenAI API usage by reusing existing embeddings

### Frontmatter Metadata Integration

The system now leverages YAML frontmatter metadata in notes to enhance semantic linking:

- Automatically extracts and parses YAML frontmatter from notes
- Incorporates metadata fields like tags, categories, and topics into similarity calculations
- Identifies and displays shared metadata fields between linked notes
- Creates more contextually relevant connections between related notes
- Handles various metadata formats including lists, strings, and numeric values

### Session Memory

The scripts use a tracking directory with JSON files to track which notes have been processed by each tool. This enables:

- Efficient processing of only new or modified notes
- Significant performance improvement for large vaults
- Complete vault coverage over multiple runs
- Tracking of note content changes through hashing

You can override this with the `--force-all` flag to process all notes regardless of their processing history.

### Link Deduplication

The system prevents duplicate links by:
- Tracking links added by each linking method
- Preventing the same note from being linked multiple times
- Providing a dedicated `--deduplicate` command to clean up existing links

## Project Roadmap

See [ROADMAP.md](ROADMAP.md) for the planned features and development timeline.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License
