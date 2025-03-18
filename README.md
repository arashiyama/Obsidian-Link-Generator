# Auto Link Obsidian

A tool for enhancing Obsidian notes with auto-tagging and linking.

## Features

- **Auto-tagging**: Automatically generate tags for notes based on content
- **Tag-based linking**: Create links between notes that share common tags
- **Semantic linking**: Create links between notes based on semantic similarity
- **GenAI linking**: Create intelligent links with AI-generated explanations
- **Note categorization**: Add visual color coding to notes for better graph organization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/auto_link_obsidian.git
cd auto_link_obsidian

# Install the package
pip install -e .
```

## Usage

### Configuration

Set your OpenAI API key in an environment variable:

```bash
export OPENAI_API_KEY="your-api-key"
```

### Running the tool

```bash
# Run all enhancement tools
obsidian-enhance --all --vault-path /path/to/your/vault

# Run specific tools
obsidian-enhance --semantic-link --vault-path /path/to/your/vault

# See all available options
obsidian-enhance --help
```

### Available Options

- `--vault-path`: Path to your Obsidian vault
- `--auto-tag`: Run auto-tagging on notes
- `--tag-link`: Run tag-based linking
- `--semantic-link`: Run semantic linking
- `--genai-link`: Run GenAI linking
- `--categorize`: Run note categorization
- `--all`: Run all enhancement tools
- `--force-all`: Process all notes (ignoring tracking)
- `--clean`: Remove all auto-generated links
- `--verbose`: Display detailed output

## Architecture

The tool is designed with a modular architecture:

- Core components:
  - Note data model
  - Configuration management
  - Embedding provider interface
  - Storage management

- Linker implementations:
  - BaseLinker abstract class
  - SemanticLinker for semantic similarity
  - TagLinker for shared tags
  - GenAILinker for AI-generated links

## Contributing

Contributions are welcome\! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI](https://openai.com/) for their powerful embedding and language models
- [Obsidian](https://obsidian.md/) for the amazing note-taking tool
