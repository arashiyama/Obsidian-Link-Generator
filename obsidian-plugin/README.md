# Auto Link Obsidian Plugin

Automatically create semantic links between notes based on content, tags, and AI-powered analysis.

## Features

- **AI-powered Semantic Linking**: Uses OpenAI embeddings to find semantically related notes
- **Customizable Similarity Threshold**: Control how similar notes need to be to create links
- **Batch Processing**: Process your entire vault or individual notes
- **YAML Frontmatter Support**: Consider metadata in similarity calculations
- **Tag Analysis**: Include tags when finding related content
- **Easy Integration**: Works with your existing Obsidian workflow

## Installation

### From the Obsidian Community Plugins

1. Open Obsidian
2. Go to Settings > Community plugins
3. Turn off "Safe mode"
4. Click "Browse" and search for "Auto Link"
5. Install the plugin and enable it

### Manual Installation

1. Download the latest release from the [GitHub releases page](https://github.com/jonathancare/auto-link-obsidian/releases)
2. Extract the ZIP file into your Obsidian vault's `.obsidian/plugins/` directory
3. Reload Obsidian
4. Enable the plugin in Settings > Community plugins

## Setup

1. After installation, go to Settings > Auto Link
2. Enter your OpenAI API key (required for generating embeddings)
3. Configure other settings to your preference

## Usage

### Generate Links for Current Note

1. Open a note you want to analyze
2. Use the command palette (Ctrl+P) and search for "Generate Semantic Links for Current Note"
3. Links to semantically related notes will be added to your note under the specified heading (default: "## Related Notes")

### Generate Links for All Notes

1. Use the command palette (Ctrl+P) and search for "Generate Semantic Links for All Notes"
2. The plugin will process all notes in your vault in batches
3. You'll see progress notifications as the plugin works through your vault

## Configuration Options

- **OpenAI API Key**: Your API key for accessing OpenAI's embedding model
- **API Endpoint**: The embedding API endpoint (default is OpenAI's)
- **Embedding Model**: The model to use for generating embeddings
- **Semantic Threshold**: Minimum similarity score (0-1) for suggesting links
- **Maximum Links**: Maximum number of links to add per note
- **Include Tags**: Consider note tags when calculating similarity
- **Include Metadata**: Consider note frontmatter when calculating similarity
- **Link Section Heading**: The heading under which links will be added
- **Processing Batch Size**: Number of notes to process in each batch
- **Show Progress Notices**: Enable/disable progress notifications
- **Debug Mode**: Enable additional logging for troubleshooting

## How It Works

Auto Link analyzes the semantic meaning of your notes using AI embeddings. It compares each note against others in your vault to find meaningful connections that might not be obvious through simple keyword matching.

When generating links, the plugin:

1. Extracts the content, tags, and metadata from the note
2. Generates an embedding using the OpenAI API
3. Compares this embedding with embeddings from other notes
4. Calculates a similarity score for each comparison
5. Adds links to the most similar notes (above your threshold)

## License

This plugin is released under the MIT License.

## Support

If you encounter any issues or have suggestions, please [open an issue](https://github.com/jonathancare/auto-link-obsidian/issues) on GitHub.
