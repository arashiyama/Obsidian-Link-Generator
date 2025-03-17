# Auto Link Obsidian: Development Roadmap

This document outlines the planned development roadmap for the Auto Link Obsidian project. It provides a high-level overview of upcoming features, improvements, and changes.

## Current Focus: Embedding Quality and Performance

- ✅ **OpenAI Embeddings Integration**: Replace TF-IDF with OpenAI embeddings for improved semantic understanding
- ✅ **Caching System**: Implement embedding cache to avoid reprocessing unchanged notes
- [ ] **Batch Processing Optimization**: Fine-tune batch sizes and parallelization for optimal performance

## Near-Term Goals (Next 3 Months)

### Enhanced Metadata Integration
- ✅ Extract and parse YAML frontmatter metadata 
- ✅ Consider metadata fields in similarity calculations
- ✅ Display shared metadata in link suggestions

### AI Note Summaries
- [ ] Create automatic summarization of note content
- [ ] Add option to include summaries in link sections
- [ ] Implement incremental summarization for changed notes

### Performance Optimizations
- [ ] Implement multithreading for non-API operations
- [ ] Add progress bars for all long-running operations
- [ ] Optimize memory usage for large vaults (1000+ notes)
- [ ] Add option to store embeddings in SQLite database

## Mid-Term Goals (3-6 Months)

### Relationship Visualization
- [ ] Export relationship data in formats compatible with visualization tools
- [ ] Create simple built-in visualization using matplotlib or plotly
- [ ] Add option to generate standalone HTML graph visualization

### Custom Link Templates
- [ ] Create a template system for different types of links
- [ ] Allow users to define custom templates in configuration
- [ ] Support markdown formatting in templates

### Advanced Tag Clustering
- [ ] Implement hierarchical clustering of related tags
- [ ] Use AI to suggest tag organization
- [ ] Create visualization of tag relationships

## Long-Term Goals (6+ Months)

### Bulk Operations Interface
- [ ] Create a CLI interface for bulk operations
- [ ] Add the ability to target specific notes or folders
- [ ] Implement a simple TUI (Text User Interface) for interactive use

### Obsidian Plugin Integration
- [ ] Research requirements for official Obsidian plugins
- [ ] Convert core functionality to JavaScript/TypeScript
- [ ] Design user-friendly settings page
- [ ] Submit to Obsidian community plugins

### Natural Language Search
- [ ] Implement embeddings-based semantic search
- [ ] Add natural language query parsing
- [ ] Create search results visualization

## Continuous Improvements

### Code Quality
- [ ] Implement comprehensive test suite
- [ ] Add type hints throughout the codebase
- [ ] Improve error handling and recovery
- [ ] Enhance logging system

### Documentation
- [ ] Create detailed API documentation
- [ ] Add more examples and tutorials
- [ ] Improve installation and setup instructions
- [ ] Add troubleshooting guide

### User Experience
- [ ] Improve console output formatting
- [ ] Add color coding for different message types
- [ ] Create a simple dashboard for monitoring processing status
- [ ] Implement a configuration wizard for first-time setup

## Community Features

### Contribution Workflow
- [ ] Create detailed contribution guidelines
- [ ] Set up templates for issues and pull requests
- [ ] Establish style guide and code standards

### Extension Points
- [ ] Design plugin architecture for custom processors
- [ ] Create hooks for pre/post processing
- [ ] Document extension API

---

This roadmap is subject to change based on user feedback, priorities, and technical considerations. Items are not necessarily in strict order of implementation.
