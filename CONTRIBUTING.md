# Contributing to Auto Link Obsidian

Thank you for considering contributing to Auto Link Obsidian! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [jonc@lacunae.org](mailto:jonc@lacunae.org).

## Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/auto_link_obsidian.git
   cd auto_link_obsidian
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

1. **Check the roadmap**: Look at the [ROADMAP.md](ROADMAP.md) file to see what features or improvements are planned.
2. **Create or choose an issue**: Either create a new issue for your feature/fix or choose an existing one to work on.
3. **Write your code**: Implement your changes following the coding standards below.
4. **Test your changes**: Make sure your changes work as expected.
5. **Commit your changes**: Use clear and descriptive commit messages.
6. **Push to your fork**: Push your branch to your GitHub fork.
7. **Create a pull request**: Submit a PR to the main repository.

## Pull Request Process

1. Ensure your code follows the project's coding standards.
2. Update documentation if necessary.
3. Include a clear description of the changes in your PR.
4. Link any related issues in your PR description using keywords like "Fixes #123" or "Relates to #456".
5. PRs need at least one approval from a maintainer before they can be merged.

## Coding Standards

- **Python Style**: Follow [PEP 8](https://pep8.org/) style guidelines.
- **Docstrings**: Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for functions and classes.
- **Type Hints**: Include Python type hints where appropriate.
- **Error Handling**: Use proper exception handling with specific exception types.
- **Comments**: Write clear comments for complex code sections.

Example of a good function definition:

```python
def extract_tags(content: str) -> list[str]:
    """
    Extract tags from content.
    
    Args:
        content: String content to extract tags from
        
    Returns:
        List of extracted tags
        
    Raises:
        ValueError: If content is empty
    """
    if not content:
        raise ValueError("Content cannot be empty")
        
    # Function implementation...
    return extracted_tags
```

## Testing

We use pytest for testing. Before submitting a PR, ensure all tests pass:

```bash
pytest
```

For new features, please add appropriate tests to cover your code.

## Documentation

- Update README.md if your changes add or modify features.
- Add or update docstrings for new or modified functions, classes, and methods.
- Consider adding examples for new features.

## Issue Reporting

When reporting issues, please include:

1. A clear and descriptive title
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Environment information (OS, Python version, etc.)
6. Screenshots or logs if applicable

## Feature Requests

Feature requests are welcome! Please provide:

1. A clear description of the feature
2. The motivation for the feature (why it would be useful)
3. Possible implementation approach (if you have ideas)

---

Thank you for contributing to Auto Link Obsidian!
