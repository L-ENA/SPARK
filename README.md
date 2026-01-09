# ⚡ SPARK

**S**ystematic **P**rotocol for **A**utomated **R**esearch **K**nowledge extraction

A Streamlit application for automated data extraction from research papers using Large Language Models. Extract structured information from titles and abstracts in RIS or CSV files using OpenAI's language models.

## Features

- **Flexible Schema Definition**: Define custom entities with examples for extraction
- **Multiple File Formats**: Support for RIS and CSV file uploads
- **Schema Management**: Save and load extraction schemas as JSON files
- **OpenAI Integration**: Use GPT-4, GPT-4-turbo, or GPT-3.5-turbo for extraction
- **Real-time Progress**: Progress bar showing extraction status
- **Results Export**: Download results as CSV with all extracted entities
- **Extraction Statistics**: View statistics on extraction success rates

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run spark/app.py
   ```

3. Follow the 4-step workflow in the app:
   - **Step 1**: Define your extraction schema or load an example
   - **Step 2**: Upload your RIS or CSV file
   - **Step 3**: Enter your OpenAI API key and select a model
   - **Step 4**: Execute extraction and download results

## Installation

### Prerequisites

- Python 3.8 or higher
- pip
- OpenAI API key

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd SPARK
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   ```bash
   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   # For production use
   pip install -r requirements.txt

   # For development (includes testing tools)
   pip install -r requirements-dev.txt
   ```

5. (Optional) Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Usage

### Running the Application

```bash
streamlit run spark/app.py
```

The app will open in your default web browser at `http://localhost:8501`.

### Using Example Data

Try the app with provided example files:

1. Load the example schema: `examples/example_schema.json`
2. Upload example data: `examples/example_data.csv` or `examples/example_data.ris`
3. Enter your OpenAI API key
4. Run extraction and download results

### Creating a Custom Schema

Schemas define what entities to extract. Example:

```json
{
  "context": "Extract key information from medical research abstracts",
  "entities": [
    {
      "name": "Disease",
      "description": "Medical conditions studied",
      "examples": ["Type 2 Diabetes", "Hypertension", "Cancer"]
    },
    {
      "name": "Intervention",
      "description": "Treatments tested",
      "examples": ["Metformin", "Exercise Program", "Surgery"]
    }
  ]
}
```

### File Format Requirements

**CSV Files:**
- Must have columns named "title" and "abstract" (case-insensitive)
- Additional columns will be preserved in output
- UTF-8 encoding recommended

**RIS Files:**
- Standard RIS format
- Fields: TY (type), TI (title), AB (abstract), AU (authors), etc.
- All metadata is preserved in output

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=spark --cov-report=html

# Run specific test file
pytest tests/test_example.py
```

### Code Formatting

```bash
# Format code with black
black spark tests

# Check code style with flake8
flake8 spark tests
```

### Type Checking

```bash
# Run mypy type checker
mypy spark
```

## Project Structure

```
SPARK/
├── spark/                      # Main package
│   ├── __init__.py            # Package initialization
│   ├── app.py                 # Streamlit application
│   └── utils.py               # Utility functions
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_example.py
│   └── test_utils.py          # Tests for utility functions
├── examples/                   # Example files
│   ├── example_schema.json    # Sample extraction schema
│   ├── example_data.csv       # Sample CSV data
│   ├── example_data.ris       # Sample RIS data
│   └── basic_usage.py         # Usage example
├── docs/                       # Documentation
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── setup.py                    # Package setup
├── pyproject.toml             # Project configuration
├── CLAUDE.md                  # AI assistant guidance
└── README.md                  # This file
```

## Important Notes

### API Costs
- OpenAI API usage incurs costs
- Cost depends on model and text length
- **gpt-4o-mini** recommended for cost-effectiveness
- Test with small datasets first

### Performance
- Each record requires a separate API call
- Processing time increases with dataset size
- For large datasets (>100 records), processing may take several minutes
- Progress bar shows real-time status

### Privacy
- API keys are stored in session state only (not saved to disk)
- Data is sent to OpenAI for processing
- Consider data privacy requirements for your use case

## Dependencies

- **streamlit**: Web application framework
- **langextract**: LLM-based entity extraction
- **pandas**: Data manipulation
- **rispy**: RIS file parsing
- **openai**: OpenAI API client

## Troubleshooting

**"CSV must contain 'title' and 'abstract' columns"**
- Check column names in your CSV file
- Column names are case-insensitive but must be spelled correctly

**"Error loading file"**
- Ensure file encoding is UTF-8
- Check file format matches selected type (RIS/CSV)

**"API key error"**
- Verify your OpenAI API key is valid
- Check API key has sufficient credits

## License

MIT License

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass with `pytest`
5. Format code with `black spark tests`
6. Submit a pull request

## Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/)
- [langextract](https://pypi.org/project/langextract/)
- [OpenAI API](https://openai.com/api/)
