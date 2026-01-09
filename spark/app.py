"""
Streamlit app for automated data extraction from research papers using LLMs
"""
import streamlit as st
import pandas as pd
import json
import textwrap
import zipfile
from io import BytesIO
from typing import Dict, List, Any
import langextract as lx
from langextract import extract
import os

# Handle imports for both package and direct execution
try:
    from spark.utils import (
        parse_ris_file,
        parse_csv_file,
        save_schema,
        load_schema,
        prepare_extraction_text,
        format_extracted_entities,
        validate_schema
    )
except ModuleNotFoundError:
    from utils import (
        parse_ris_file,
        parse_csv_file,
        save_schema,
        load_schema,
        prepare_extraction_text,
        format_extracted_entities,
        validate_schema
    )


def initialize_session_state():
    """Initialize session state variables"""
    if 'schema' not in st.session_state:
        st.session_state.schema = {
            'context': '',
            'prompt_description': '',
            'entities': []
        }
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
    if 'model' not in st.session_state:
        st.session_state.model = 'gpt-4o-mini'
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'loaded_schema_file' not in st.session_state:
        st.session_state.loaded_schema_file = None
    if 'html_visualizations' not in st.session_state:
        st.session_state.html_visualizations = []


def render_step1_schema_definition():
    """Step 1: Define extraction schema"""
    st.header("Step 1: Define Data Extraction Schema")

    st.markdown("""
    Define a labeled example and the entities you want to extract from your research papers.
    The Extraction Context should contain an example title and abstract from which you'll extract entities.
    """)

    # Schema save/load functionality
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Load Existing Schema")
        uploaded_schema = st.file_uploader(
            "Upload schema JSON file",
            type=['json'],
            key='schema_upload'
        )

        if uploaded_schema is not None:
            # Create unique identifier for the uploaded file
            file_id = f"{uploaded_schema.name}_{uploaded_schema.size}"

            # Only process if this is a new/different file
            if file_id != st.session_state.loaded_schema_file:
                try:
                    schema_content = uploaded_schema.read().decode('utf-8')
                    loaded_schema = json.loads(schema_content)
                    is_valid, error_msg = validate_schema(loaded_schema)

                    if is_valid:
                        st.session_state.schema = loaded_schema
                        # Ensure prompt_description exists in loaded schema
                        if 'prompt_description' not in st.session_state.schema:
                            st.session_state.schema['prompt_description'] = ''
                        st.session_state.loaded_schema_file = file_id
                        st.success("Schema loaded successfully!")
                        st.rerun()
                    else:
                        st.error(f"Invalid schema: {error_msg}")
                except Exception as e:
                    st.error(f"Error loading schema: {str(e)}")

    with col2:
        st.subheader("Save Current Schema")
        if st.button("Download Schema as JSON"):
            if st.session_state.schema['entities']:
                # Update prompt_description before saving
                entity_names = [entity['name'] for entity in st.session_state.schema['entities']]
                entity_list = ', '.join(entity_names)
                st.session_state.schema['prompt_description'] = textwrap.dedent(f"""\
                    Extract {entity_list} in order of appearance.
                    Use exact text for extractions. Do not paraphrase or overlap entities.
                    Provide meaningful attributes for each entity to add context.""")

                schema_json = json.dumps(st.session_state.schema, indent=2)
                st.download_button(
                    label="Download",
                    data=schema_json,
                    file_name="extraction_schema.json",
                    mime="application/json"
                )
            else:
                st.warning("Please define at least one entity before saving")

    st.divider()

    # Extraction Context - example text
    st.subheader("Extraction Context")
    st.markdown("""
    Provide an example title and abstract from a research paper.
    This serves as a labeled example for the extraction task.
    """)
    context = st.text_area(
        "Example Title and Abstract",
        value=st.session_state.schema.get('context', ''),
        height=200,
        help="Paste the title and abstract of a research paper that contains examples of the entities you want to extract",
        placeholder="Title: Example Study on Treatment Effectiveness\n\nAbstract: This randomized controlled trial evaluated the effectiveness of..."
    )
    st.session_state.schema['context'] = context

    st.divider()

    # Dynamic Prompt Description - non-editable, updates with entity names
    if st.session_state.schema['entities']:
        entity_names = [entity['name'] for entity in st.session_state.schema['entities']]
        entity_list = ', '.join(entity_names)
        prompt_description = textwrap.dedent(f"""\
            Extract {entity_list} in order of appearance.
            Use exact text for extractions. Do not paraphrase or overlap entities.
            Provide meaningful attributes for each entity to add context.""")

        # Update in schema
        st.session_state.schema['prompt_description'] = prompt_description

        # Display as info box
        st.info(f"**Extraction Instructions:**\n\n{prompt_description}")
        st.divider()

    # Entity definitions
    st.subheader("Entity Definitions")
    st.markdown("""
    Define entities to extract. Examples should be taken directly from the Extraction Context above.
    """)

    # Display existing entities
    if st.session_state.schema['entities']:
        for idx, entity in enumerate(st.session_state.schema['entities']):
            with st.expander(f"Entity {idx + 1}: {entity['name']}", expanded=False):
                col1, col2 = st.columns([4, 1])

                with col1:
                    entity_name = st.text_input(
                        "Entity Name",
                        value=entity['name'],
                        key=f"entity_name_{idx}"
                    )
                    entity_description = st.text_area(
                        "Description (optional)",
                        value=entity.get('description', ''),
                        key=f"entity_desc_{idx}",
                        height=100
                    )
                    examples_text = st.text_area(
                        "Examples (one per line, from Extraction Context)",
                        value='\n'.join(entity['examples']),
                        key=f"entity_examples_{idx}",
                        height=120,
                        help="Provide example values for this entity type, taken directly from the Extraction Context above"
                    )

                with col2:
                    if st.button("Delete", key=f"delete_{idx}"):
                        st.session_state.schema['entities'].pop(idx)
                        st.rerun()

                # Update entity
                st.session_state.schema['entities'][idx] = {
                    'name': entity_name,
                    'description': entity_description,
                    'examples': [ex.strip() for ex in examples_text.split('\n') if ex.strip()]
                }

    # Add new entity
    st.subheader("Add New Entity")
    with st.form("add_entity_form", clear_on_submit=True):
        new_entity_name = st.text_input("Entity Name", placeholder="e.g., Disease, Intervention, Outcome")
        new_entity_description = st.text_area(
            "Description (optional)",
            placeholder="Describe what this entity represents",
            height=100
        )
        new_entity_examples = st.text_area(
            "Examples (one per line, from Extraction Context)",
            placeholder="Example 1 (from context above)\nExample 2 (from context above)\nExample 3 (from context above)",
            height=120,
            help="Provide examples taken directly from the Extraction Context"
        )

        submitted = st.form_submit_button("Add Entity")

        if submitted and new_entity_name:
            new_entity = {
                'name': new_entity_name,
                'description': new_entity_description,
                'examples': [ex.strip() for ex in new_entity_examples.split('\n') if ex.strip()]
            }
            st.session_state.schema['entities'].append(new_entity)
            st.success(f"Added entity: {new_entity_name}")
            st.rerun()

    # Show schema summary
    if st.session_state.schema['entities']:
        st.success(f"✓ Schema defined with {len(st.session_state.schema['entities'])} entities")


def render_step2_file_upload():
    """Step 2: Upload data file"""
    st.header("Step 2: Upload Data File")

    st.markdown("""
    Upload a RIS file or CSV file containing research paper titles and abstracts.
    For CSV files, ensure columns are named 'Title' and 'Abstract'.
    """)

    file_type = st.radio(
        "Select file type",
        ["RIS File", "CSV File"],
        horizontal=True
    )

    if file_type == "RIS File":
        uploaded_file = st.file_uploader(
            "Upload RIS file",
            type=['ris', 'txt'],
            key='ris_upload'
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            key='csv_upload'
        )

    if uploaded_file is not None:
        try:
            file_content = uploaded_file.read().decode('utf-8')

            if file_type == "RIS File":
                df = parse_ris_file(file_content)
            else:
                df = parse_csv_file(file_content)

            st.session_state.uploaded_data = df

            st.success(f"✓ File loaded successfully! Found {len(df)} records.")

            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Show statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                has_title = df['title'].notna().sum()
                st.metric("Records with Title", has_title)
            with col3:
                has_abstract = df['abstract'].notna().sum()
                st.metric("Records with Abstract", has_abstract)

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.session_state.uploaded_data = None


def render_step3_api_config():
    """Step 3: Configure API and model"""
    st.header("Step 3: Configure API and Model")

    st.markdown("""
    Enter your OpenAI API key and select the model to use for extraction.
    """)

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.api_key,
        help="Your API key will not be stored permanently"
    )
    st.session_state.api_key = api_key

    # Model selection
    model_options = [
        'gpt-4o',
        'gpt-4o-mini',
        'gpt-4-turbo',
        'gpt-4',
        'gpt-3.5-turbo'
    ]

    selected_model = st.selectbox(
        "Select Model",
        options=model_options,
        index=model_options.index(st.session_state.model),
        help="gpt-4o-mini is recommended for cost-effectiveness"
    )
    st.session_state.model = selected_model

    if api_key:
        st.success("✓ API key provided")
    else:
        st.warning("Please enter your OpenAI API key to proceed")


def render_step4_execution():
    """Step 4: Execute extraction"""
    st.header("Step 4: Execute Extraction")

    # Check prerequisites
    ready_to_run = True
    issues = []

    if not st.session_state.schema['entities']:
        ready_to_run = False
        issues.append("No entities defined in schema")

    if st.session_state.uploaded_data is None:
        ready_to_run = False
        issues.append("No data file uploaded")

    if not st.session_state.api_key:
        ready_to_run = False
        issues.append("No API key provided")

    if not ready_to_run:
        st.warning("Please complete the previous steps:")
        for issue in issues:
            st.write(f"- {issue}")
        return

    # Show extraction summary
    st.subheader("Extraction Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Records to Process", len(st.session_state.uploaded_data))
    with col2:
        st.metric("Entities to Extract", len(st.session_state.schema['entities']))
    with col3:
        st.metric("Model", st.session_state.model)

    st.divider()

    # Execution button
    if st.button("Start Extraction", type="primary", use_container_width=True):
        extract_data()

    # Show results if available
    if st.session_state.results is not None:
        st.success("✓ Extraction completed!")

        # Show results preview
        st.subheader("Results Preview")
        st.dataframe(st.session_state.results.head(10), use_container_width=True)

        # Download button
        csv = st.session_state.results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="extraction_results.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )

        # Show extraction statistics
        st.subheader("Extraction Statistics")
        entity_names = [e['name'] for e in st.session_state.schema['entities']]

        stats_data = []
        for entity_name in entity_names:
            if entity_name in st.session_state.results.columns:
                non_empty = st.session_state.results[entity_name].notna().sum()
                has_values = (st.session_state.results[entity_name] != '').sum()
                stats_data.append({
                    'Entity': entity_name,
                    'Records with Extractions': has_values,
                    'Percentage': f"{(has_values / len(st.session_state.results) * 100):.1f}%"
                })

        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)

        # Show HTML visualizations download section
        if st.session_state.html_visualizations:
            st.divider()
            st.subheader("Interactive HTML Visualizations")
            st.markdown("""
            Download interactive HTML files to view extraction results with clickable entity highlighting.
            Each file shows the original text with buttons to highlight different entity types.
            """)

            # Display download buttons in a grid
            num_files = len(st.session_state.html_visualizations)
            st.write(f"**{num_files} visualization(s) available**")

            # Show first few files with individual download buttons
            max_display = 10
            if num_files <= max_display:
                # Show all files individually
                cols = st.columns(min(5, num_files))
                for i, html_file in enumerate(st.session_state.html_visualizations):
                    with cols[i % 5]:
                        st.download_button(
                            label=f"📄 Record {i + 1}",
                            data=html_file['content'],
                            file_name=html_file['filename'],
                            mime="text/html",
                            use_container_width=True
                        )
            else:
                # Show first 10 and provide option to download all as zip
                st.write("Showing first 10 records:")
                cols = st.columns(5)
                for i in range(min(10, num_files)):
                    html_file = st.session_state.html_visualizations[i]
                    with cols[i % 5]:
                        st.download_button(
                            label=f"📄 Record {i + 1}",
                            data=html_file['content'],
                            file_name=html_file['filename'],
                            mime="text/html",
                            use_container_width=True
                        )

                # Option to download all as zip
                st.divider()
                st.markdown("**Download All Visualizations**")

                # Create zip file in memory
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for html_file in st.session_state.html_visualizations:
                        zip_file.writestr(html_file['filename'], html_file['content'])

                st.download_button(
                    label=f"⬇️ Download All {num_files} HTML Files (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="extraction_visualizations.zip",
                    mime="application/zip",
                    type="primary",
                    use_container_width=True
                )


def generate_interactive_html(record_number: int, text: str, result: Any, entity_names: List[str]) -> str:
    """
    Generate an interactive HTML visualization for extraction results.

    Args:
        record_number: Sequential number for the record
        text: Original text that was processed
        result: AnnotatedDocument result from langextract
        entity_names: List of entity type names

    Returns:
        HTML string with interactive visualization
    """
    # Define colors for different entity types (cycling through if more entities than colors)
    colors = [
        '#FFB6C1', '#87CEEB', '#90EE90', '#FFD700', '#DDA0DD',
        '#FFA07A', '#98FB98', '#DEB887', '#F0E68C', '#E0BBE4'
    ]

    # Group extractions by entity class
    entity_groups = {}
    for extraction in result.extractions:
        entity_class = extraction.extraction_class
        if entity_class not in entity_groups:
            entity_groups[entity_class] = []
        entity_groups[entity_class].append({
            'text': extraction.extraction_text,
            'start': extraction.char_interval.start_pos,
            'end': extraction.char_interval.end_pos
        })

    # Create color mapping for entity types
    entity_colors = {name: colors[i % len(colors)] for i, name in enumerate(entity_names)}

    # Build JavaScript data structure
    js_entities = {}
    for entity_name in entity_names:
        if entity_name in entity_groups:
            js_entities[entity_name] = entity_groups[entity_name]
        else:
            js_entities[entity_name] = []

    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Extraction Result - Record {record_number}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
        }}
        .controls {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .button-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .entity-button {{
            padding: 10px 20px;
            border: 2px solid #ddd;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s;
            background-color: white;
        }}
        .entity-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .entity-button.active {{
            border-width: 3px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .text-container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            line-height: 1.8;
            font-size: 16px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .highlight {{
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: 500;
            transition: all 0.2s;
        }}
        .stats {{
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats-item {{
            display: inline-block;
            margin-right: 20px;
            font-size: 14px;
        }}
        .stats-label {{
            font-weight: bold;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Extraction Results - Record {record_number}</h1>
    </div>

    <div class="stats">
        <div class="stats-item">
            <span class="stats-label">Total Extractions:</span>
            <span id="total-extractions">{len(result.extractions)}</span>
        </div>
        <div class="stats-item">
            <span class="stats-label">Entity Types:</span>
            <span>{len([k for k, v in js_entities.items() if v])}</span>
        </div>
    </div>

    <div class="controls">
        <h3 style="margin-top: 0;">Entity Types (Click to Highlight)</h3>
        <div class="button-group" id="button-group">
            <!-- Buttons will be added by JavaScript -->
        </div>
    </div>

    <div class="text-container" id="text-container">
        {text}
    </div>

    <script>
        const entities = {json.dumps(js_entities)};
        const colors = {json.dumps(entity_colors)};
        const originalText = {json.dumps(text)};

        let activeEntity = null;

        function createButtons() {{
            const buttonGroup = document.getElementById('button-group');

            for (const [entityType, extractions] of Object.entries(entities)) {{
                if (extractions.length > 0) {{
                    const button = document.createElement('button');
                    button.className = 'entity-button';
                    button.textContent = `${{entityType}} (${{extractions.length}})`;
                    button.style.backgroundColor = colors[entityType];
                    button.style.borderColor = colors[entityType];
                    button.onclick = () => toggleHighlight(entityType, button);
                    buttonGroup.appendChild(button);
                }}
            }}
        }}

        function toggleHighlight(entityType, button) {{
            const allButtons = document.querySelectorAll('.entity-button');

            if (activeEntity === entityType) {{
                // Deactivate
                activeEntity = null;
                button.classList.remove('active');
                showOriginalText();
            }} else {{
                // Activate
                activeEntity = entityType;
                allButtons.forEach(b => b.classList.remove('active'));
                button.classList.add('active');
                highlightEntities(entityType);
            }}
        }}

        function showOriginalText() {{
            document.getElementById('text-container').textContent = originalText;
        }}

        function highlightEntities(entityType) {{
            const extractions = entities[entityType];
            const color = colors[entityType];

            // Sort extractions by start position (descending) to avoid position shift
            const sortedExtractions = [...extractions].sort((a, b) => b.start - a.start);

            let highlightedText = originalText;

            for (const extraction of sortedExtractions) {{
                const before = highlightedText.substring(0, extraction.start);
                const match = highlightedText.substring(extraction.start, extraction.end);
                const after = highlightedText.substring(extraction.end);

                const highlighted = `<span class="highlight" style="background-color: ${{color}}; border: 2px solid ${{darkenColor(color)}};">${{escapeHtml(match)}}</span>`;
                highlightedText = before + highlighted + after;
            }}

            document.getElementById('text-container').innerHTML = highlightedText;
        }}

        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}

        function darkenColor(color) {{
            // Simple color darkening
            const hex = color.replace('#', '');
            const r = Math.max(0, parseInt(hex.substr(0, 2), 16) - 40);
            const g = Math.max(0, parseInt(hex.substr(2, 2), 16) - 40);
            const b = Math.max(0, parseInt(hex.substr(4, 2), 16) - 40);
            return `#${{r.toString(16).padStart(2, '0')}}${{g.toString(16).padStart(2, '0')}}${{b.toString(16).padStart(2, '0')}}`;
        }}

        // Initialize
        createButtons();
    </script>
</body>
</html>
"""

    return html_template


def extract_data():
    """Execute the extraction process"""
    df = st.session_state.uploaded_data.copy()
    schema = st.session_state.schema
    api_key = st.session_state.api_key
    model = st.session_state.model

    # Set API key as environment variable
    os.environ['OPENAI_API_KEY'] = api_key

    # Prepare entity schema for langextract
    entity_names = [entity['name'] for entity in schema['entities']]

    # Create examples in the format required by langextract
    # Build a list of Extraction objects from all entities' examples
    extractions = []
    for entity in schema['entities']:
        entity_name = entity['name']
        entity_description=entity["description"]
        for example_text in entity['examples']:
            extractions.append(
                lx.data.Extraction(
                    extraction_class=entity_name,
                    extraction_text=example_text,
                    description=entity_description,
                    attributes={}
                )
            )

    # Create one ExampleData object with the context text and all extractions
    examples = [
        lx.data.ExampleData(
            text=schema['context'],
            extractions=extractions
        )
    ]

    # Initialize result columns
    for entity_name in entity_names:
        df[entity_name] = ''

    # Clear previous HTML visualizations
    st.session_state.html_visualizations = []

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_records = len(df)

    # Process each record
    for idx, row in df.iterrows():
        progress = (idx + 1) / total_records
        progress_bar.progress(progress)
        status_text.text(f"Processing record {idx + 1} of {total_records}")

        # Prepare text for extraction
        text = prepare_extraction_text(row).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

        if not text:
            continue

        try:
            # Perform extraction using langextract
            result = extract(
                text_or_documents=text,
                examples=examples,
                prompt_description=schema['prompt_description'],
                model_id=model
            )

            # Process AnnotatedDocument result
            # Group extractions by entity class (extraction_class)
            entity_extractions = {entity_name: [] for entity_name in entity_names}

            for extraction in result.extractions:
                entity_class = extraction.extraction_class
                if entity_class in entity_extractions:
                    entity_extractions[entity_class].append(extraction.extraction_text)

            # Format and store results
            for entity_name in entity_names:
                extracted_texts = entity_extractions[entity_name]
                if extracted_texts:
                    # Remove duplicates and join with semicolon
                    unique_entities = sorted(set(extracted_texts))
                    df.at[idx, entity_name] = '; '.join(unique_entities)

            # Generate interactive HTML visualization
            html_content = generate_interactive_html(
                record_number=idx + 1,
                text=text,
                result=result,
                entity_names=entity_names
            )
            st.session_state.html_visualizations.append({
                'filename': f'{idx + 1}.html',
                'content': html_content
            })

        except Exception as e:
            st.warning(f"Error processing record {idx + 1}: {str(e)}")
            continue

    progress_bar.progress(1.0)
    status_text.text(f"Completed! Processed {total_records} records.")

    # Store results
    st.session_state.results = df


def main():
    """Main application"""
    st.set_page_config(
        page_title="SPARK - Data Extraction",
        page_icon="⚡",
        layout="wide"
    )

    st.title("⚡ SPARK - LLM-Powered Data Extraction")
    st.markdown("""
    Automate verbatim data extraction from research paper titles and abstracts using Large Language Models.
    """)

    # Initialize session state
    initialize_session_state()

    # Create tabs for each step
    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ Define Schema",
        "2️⃣ Upload Data",
        "3️⃣ Configure API",
        "4️⃣ Execute"
    ])

    with tab1:
        render_step1_schema_definition()

    with tab2:
        render_step2_file_upload()

    with tab3:
        render_step3_api_config()

    with tab4:
        render_step4_execution()


if __name__ == "__main__":
    main()
