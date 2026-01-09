"""
Utility functions for data extraction app
"""
import json
import pandas as pd
import rispy
from typing import Dict, List, Any, Tuple
from io import StringIO


def parse_ris_file(file_content: str) -> pd.DataFrame:
    """
    Parse RIS file content and return a DataFrame.

    Args:
        file_content: String content of RIS file

    Returns:
        DataFrame with columns: title, abstract, and other metadata
    """
    # Parse RIS content
    entries = rispy.loads(file_content)

    # Convert to list of dictionaries
    records = []
    for entry in entries:
        record = {
            'title': entry.get('title', entry.get('primary_title', '')),
            'abstract': entry.get('abstract', ''),
            'authors': '; '.join(entry.get('authors', [])) if entry.get('authors') else '',
            'year': entry.get('year', ''),
            'journal': entry.get('journal_name', entry.get('secondary_title', '')),
            'doi': entry.get('doi', ''),
            'keywords': '; '.join(entry.get('keywords', [])) if entry.get('keywords') else '',
            'type': entry.get('type_of_reference', ''),
        }
        records.append(record)

    return pd.DataFrame(records)


def parse_csv_file(file_content: str) -> pd.DataFrame:
    """
    Parse CSV file content and return a DataFrame.
    Expects columns 'title' and 'abstract' (case-insensitive).

    Args:
        file_content: String content of CSV file

    Returns:
        DataFrame with standardized column names
    """
    df = pd.read_csv(StringIO(file_content))

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()

    # Check for required columns
    if 'title' not in df.columns or 'abstract' not in df.columns:
        raise ValueError("CSV must contain 'title' and 'abstract' columns")

    return df


def save_schema(schema: Dict[str, Any], filepath: str) -> None:
    """
    Save extraction schema to JSON file.

    Args:
        schema: Dictionary containing context and entities
        filepath: Path to save the schema
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)


def load_schema(filepath: str) -> Dict[str, Any]:
    """
    Load extraction schema from JSON file.

    Args:
        filepath: Path to the schema file

    Returns:
        Dictionary containing context and entities
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_extraction_text(row: pd.Series) -> str:
    """
    Prepare text for extraction from a row.

    Args:
        row: DataFrame row containing title and abstract

    Returns:
        Combined text for extraction
    """
    title = str(row.get('title', ''))
    abstract = str(row.get('abstract', ''))

    if title and abstract:
        return f"Title: {title}\n\nAbstract: {abstract}"
    elif title:
        return f"Title: {title}"
    elif abstract:
        return f"Abstract: {abstract}"
    else:
        return ""


def format_extracted_entities(entities: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Format extracted entities as semicolon-separated strings.

    Args:
        entities: Dictionary mapping entity types to lists of entities

    Returns:
        Dictionary mapping entity types to semicolon-separated strings
    """
    return {
        entity_type: '; '.join(sorted(set(values))) if values else ''
        for entity_type, values in entities.items()
    }


def validate_schema(schema: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate extraction schema structure.

    Args:
        schema: Schema dictionary to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(schema, dict):
        return False, "Schema must be a dictionary"

    if 'context' not in schema:
        return False, "Schema must contain 'context' field"

    if 'entities' not in schema or not isinstance(schema['entities'], list):
        return False, "Schema must contain 'entities' list"

    if len(schema['entities']) == 0:
        return False, "Schema must contain at least one entity"

    for entity in schema['entities']:
        if not isinstance(entity, dict):
            return False, "Each entity must be a dictionary"

        if 'name' not in entity:
            return False, "Each entity must have a 'name' field"

        if 'examples' not in entity or not isinstance(entity['examples'], list):
            return False, "Each entity must have an 'examples' list"

    return True, ""
