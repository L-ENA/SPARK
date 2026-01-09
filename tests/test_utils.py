"""
Tests for utility functions
"""
import pytest
import pandas as pd
import json
import tempfile
import os
from spark.utils import (
    parse_ris_file,
    parse_csv_file,
    save_schema,
    load_schema,
    prepare_extraction_text,
    format_extracted_entities,
    validate_schema
)


def test_parse_csv_file():
    """Test CSV file parsing"""
    csv_content = """title,abstract
"Test Title","Test Abstract"
"Another Title","Another Abstract"
"""
    df = parse_csv_file(csv_content)

    assert len(df) == 2
    assert 'title' in df.columns
    assert 'abstract' in df.columns
    assert df.iloc[0]['title'] == "Test Title"


def test_parse_csv_file_case_insensitive():
    """Test CSV parsing with different column name cases"""
    csv_content = """Title,Abstract
"Test Title","Test Abstract"
"""
    df = parse_csv_file(csv_content)

    assert 'title' in df.columns
    assert 'abstract' in df.columns


def test_parse_csv_file_missing_columns():
    """Test CSV parsing with missing required columns"""
    csv_content = """title,other
"Test Title","Other Data"
"""
    with pytest.raises(ValueError, match="must contain 'title' and 'abstract'"):
        parse_csv_file(csv_content)


def test_save_and_load_schema():
    """Test schema save and load functionality"""
    schema = {
        'context': 'Test context',
        'entities': [
            {
                'name': 'TestEntity',
                'description': 'Test description',
                'examples': ['example1', 'example2']
            }
        ]
    }

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name

    try:
        # Save schema
        save_schema(schema, temp_path)

        # Load schema
        loaded_schema = load_schema(temp_path)

        # Verify
        assert loaded_schema == schema
        assert loaded_schema['context'] == 'Test context'
        assert len(loaded_schema['entities']) == 1
        assert loaded_schema['entities'][0]['name'] == 'TestEntity'
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_prepare_extraction_text():
    """Test extraction text preparation"""
    row = pd.Series({
        'title': 'Test Title',
        'abstract': 'Test Abstract'
    })

    text = prepare_extraction_text(row)

    assert 'Title: Test Title' in text
    assert 'Abstract: Test Abstract' in text


def test_prepare_extraction_text_title_only():
    """Test extraction text with only title"""
    row = pd.Series({
        'title': 'Test Title',
        'abstract': None
    })

    text = prepare_extraction_text(row)

    assert text == 'Title: Test Title'


def test_prepare_extraction_text_abstract_only():
    """Test extraction text with only abstract"""
    row = pd.Series({
        'title': None,
        'abstract': 'Test Abstract'
    })

    text = prepare_extraction_text(row)

    assert text == 'Abstract: Test Abstract'


def test_format_extracted_entities():
    """Test entity formatting"""
    entities = {
        'Disease': ['Diabetes', 'Hypertension', 'Diabetes'],
        'Intervention': ['Drug A', 'Drug B'],
        'Empty': []
    }

    formatted = format_extracted_entities(entities)

    assert formatted['Disease'] == 'Diabetes; Hypertension'
    assert formatted['Intervention'] == 'Drug A; Drug B'
    assert formatted['Empty'] == ''


def test_validate_schema_valid():
    """Test schema validation with valid schema"""
    schema = {
        'context': 'Test context',
        'entities': [
            {
                'name': 'Entity1',
                'examples': ['ex1', 'ex2']
            }
        ]
    }

    is_valid, error_msg = validate_schema(schema)

    assert is_valid is True
    assert error_msg == ''


def test_validate_schema_missing_context():
    """Test schema validation with missing context"""
    schema = {
        'entities': [
            {
                'name': 'Entity1',
                'examples': ['ex1']
            }
        ]
    }

    is_valid, error_msg = validate_schema(schema)

    assert is_valid is False
    assert 'context' in error_msg


def test_validate_schema_missing_entities():
    """Test schema validation with missing entities"""
    schema = {
        'context': 'Test context'
    }

    is_valid, error_msg = validate_schema(schema)

    assert is_valid is False
    assert 'entities' in error_msg


def test_validate_schema_empty_entities():
    """Test schema validation with empty entities list"""
    schema = {
        'context': 'Test context',
        'entities': []
    }

    is_valid, error_msg = validate_schema(schema)

    assert is_valid is False
    assert 'at least one entity' in error_msg


def test_validate_schema_invalid_entity():
    """Test schema validation with invalid entity structure"""
    schema = {
        'context': 'Test context',
        'entities': [
            {
                'name': 'Entity1'
                # Missing 'examples' field
            }
        ]
    }

    is_valid, error_msg = validate_schema(schema)

    assert is_valid is False
    assert 'examples' in error_msg


def test_parse_ris_file():
    """Test RIS file parsing"""
    ris_content = """TY  - JOUR
TI  - Test Article
AU  - Smith, John
PY  - 2023
AB  - This is a test abstract
KW  - keyword1
KW  - keyword2
ER  -
"""
    df = parse_ris_file(ris_content)

    assert len(df) == 1
    assert 'title' in df.columns
    assert 'abstract' in df.columns
    assert df.iloc[0]['title'] == 'Test Article'
    assert df.iloc[0]['abstract'] == 'This is a test abstract'
    assert 'Smith, John' in df.iloc[0]['authors']
