# Contributing to Fundus QA System

Thank you for your interest in contributing to the Fundus Disease QA System!

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear description of the problem
- Steps to reproduce the bug
- Expected behavior vs. actual behavior
- Your environment (OS, Python version, GPU specs)

### Suggesting Enhancements

We welcome suggestions for improvements:
- New retrieval strategies
- Better anti-hallucination mechanisms
- Additional evaluation metrics
- Performance optimizations

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## Development Setup

```bash
# Clone the repository
git clone https://github.com/your-username/low-resource-fundus-qa.git
cd low-resource-fundus-qa

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests (if available)
pytest tests/
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and modular

## Documentation

- Update README.md for user-facing changes
- Add inline comments for complex logic
- Update EXAMPLES.md for new usage patterns

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
