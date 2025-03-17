# Auto Link Obsidian Tests

This directory contains unit tests for the Auto Link Obsidian project.

## Running Tests

You can run all tests using the test runner script at the project root:

```bash
python run_tests.py
```

Or you can run individual test files directly:

```bash
python -m tests.test_utils
python -m tests.test_semantic_linker
```

## Test Structure

The tests are organized as follows:

- `test_utils.py`: Tests for utility functions in utils.py
- `test_semantic_linker.py`: Tests for semantic_linker.py including frontmatter extraction and embedding functions

## Writing New Tests

When adding new tests:

1. Create a new file named `test_<module_name>.py` for the module you're testing
2. Follow the unittest framework pattern:
   - Import unittest and the module being tested
   - Create a class that extends unittest.TestCase
   - Write methods that start with 'test_' to test individual functions
   - Use assertions to validate expected behavior
3. Consider using mocks for external dependencies like API calls
4. Ensure your tests are isolated and don't depend on external state

## Test Coverage

The goal is to achieve high test coverage for all critical components:

- Utility functions
- Parsing and content manipulation
- Semantic linking logic
- Embedding generation and handling
- Caching mechanisms
- Error handling

## Example Test

Here's a simple example of a test:

```python
def test_generate_note_hash(self):
    """Test generating a content hash."""
    content1 = "Test content"
    content2 = "Test content"
    content3 = "Different content"
    
    hash1 = utils.generate_note_hash(content1)
    hash2 = utils.generate_note_hash(content2)
    hash3 = utils.generate_note_hash(content3)
    
    self.assertEqual(hash1, hash2)  # Same content should have same hash
    self.assertNotEqual(hash1, hash3)  # Different content should have different hash
