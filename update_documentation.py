#!/usr/bin/env python3
"""
Script to update documentation with Environment Setup section
"""

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def add_code_block(doc, code_text):
    """Add a code block with monospace font and gray background."""
    para = doc.add_paragraph()
    run = para.add_run(code_text)
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    para.paragraph_format.left_indent = Inches(0.5)
    para.paragraph_format.space_before = Pt(6)
    para.paragraph_format.space_after = Pt(6)
    return para

def add_heading_with_style(doc, text, level=1):
    """Add a styled heading to the document."""
    heading = doc.add_heading(text, level=level)
    if level == 1:
        heading.runs[0].font.color.rgb = RGBColor(0, 51, 102)
    return heading

def insert_environment_setup(doc):
    """Insert Environment Setup section after Table of Contents."""

    # Find the first page break (after TOC)
    for i, para in enumerate(doc.paragraphs):
        if 'Table of Contents' in para.text:
            # Find the TOC list end
            toc_end_idx = i
            for j in range(i+1, len(doc.paragraphs)):
                if doc.paragraphs[j].text.strip() == '' or 'Overview' in doc.paragraphs[j].text:
                    toc_end_idx = j
                    break

            # Update TOC to include new section
            # Find "1. Overview" and insert before it
            for idx in range(i, min(i+15, len(doc.paragraphs))):
                if '1. Overview' in doc.paragraphs[idx].text:
                    # Insert new TOC item
                    new_para = doc.paragraphs[idx].insert_paragraph_before('1. Environment Setup')
                    new_para.style = 'List Number'

                    # Update numbering of subsequent items
                    for update_idx in range(idx+1, min(idx+12, len(doc.paragraphs))):
                        text = doc.paragraphs[update_idx].text
                        if text.startswith('1. Overview'):
                            doc.paragraphs[update_idx].text = '2. Overview'
                        elif text.startswith('2. Key Features'):
                            doc.paragraphs[update_idx].text = '3. Key Features'
                        elif text.startswith('3. Architecture'):
                            doc.paragraphs[update_idx].text = '4. Architecture & Design'
                        elif text.startswith('4. Configuration'):
                            doc.paragraphs[update_idx].text = '5. Configuration'
                        elif text.startswith('5. Main Components'):
                            doc.paragraphs[update_idx].text = '6. Main Components'
                        elif text.startswith('6. Transfer Methods'):
                            doc.paragraphs[update_idx].text = '7. Transfer Methods'
                        elif text.startswith('7. Usage Instructions'):
                            doc.paragraphs[update_idx].text = '8. Usage Instructions'
                        elif text.startswith('8. Logging'):
                            doc.paragraphs[update_idx].text = '9. Logging & Monitoring'
                        elif text.startswith('9. Error Handling'):
                            doc.paragraphs[update_idx].text = '10. Error Handling'
                        elif text.startswith('10. Best Practices'):
                            doc.paragraphs[update_idx].text = '11. Best Practices'
                    break
            break

    # Find where to insert the Environment Setup section (after first page break, before "1. Overview" or "2. Overview")
    insert_position = None
    for i, para in enumerate(doc.paragraphs):
        if '2. Overview' in para.text or ('1. Overview' in para.text and i > 10):
            insert_position = i
            break

    if insert_position is None:
        print("Could not find insertion point, appending to end")
        return

    # Insert section content at the found position
    # We need to insert in reverse order since each insert_paragraph_before shifts indices

    # Get the paragraph to insert before
    anchor_para = doc.paragraphs[insert_position]

    # Create content paragraphs (will be inserted in reverse)
    content = []

    # Section heading
    p = anchor_para.insert_paragraph_before()
    p.text = '1. Environment Setup'
    p.style = 'Heading 1'
    p.runs[0].font.color.rgb = RGBColor(0, 51, 102)

    # Introduction
    p = anchor_para.insert_paragraph_before()
    p.text = ('Before running the SharePoint TAR to Azure Blob transfer script, you need to set up your '
              'Python environment and install the required dependencies.')

    # 1.1 Prerequisites
    p = anchor_para.insert_paragraph_before()
    p.text = '1.1 Prerequisites'
    p.style = 'Heading 2'

    prereqs = [
        'Python 3.8 or higher',
        'pip (Python package installer)',
        'Azure Storage Account with Blob Storage',
        'SharePoint site with appropriate access permissions',
        'Azure AD App Registration with client credentials',
    ]
    for prereq in prereqs:
        p = anchor_para.insert_paragraph_before(prereq, style='List Bullet')

    # 1.2 Installing Dependencies
    p = anchor_para.insert_paragraph_before()
    p.text = '1.2 Installing Dependencies'
    p.style = 'Heading 2'

    p = anchor_para.insert_paragraph_before()
    p.text = 'The script requires several Python packages. You can install them using pip:'

    # Code block for pip install
    p = anchor_para.insert_paragraph_before()
    run = p.add_run('pip install azure-storage-blob>=12.19.0 PyYAML>=6.0 requests')
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)

    # Individual package descriptions
    p = anchor_para.insert_paragraph_before()
    p.text = 'Core dependencies:'

    deps = [
        ('azure-storage-blob', 'Microsoft Azure Blob Storage client library for Python'),
        ('azure-core', 'Core library for Azure SDK (installed automatically with azure-storage-blob)'),
        ('PyYAML', 'YAML parser for reading configuration files'),
        ('requests', 'HTTP library for API calls and streaming transfers'),
    ]

    for dep_name, dep_desc in deps:
        p = anchor_para.insert_paragraph_before()
        p.add_run(f'{dep_name}: ').bold = True
        p.add_run(dep_desc)

    # 1.3 Project Requirements
    p = anchor_para.insert_paragraph_before()
    p.text = '1.3 Installing from requirements.txt'
    p.style = 'Heading 2'

    p = anchor_para.insert_paragraph_before()
    p.text = ('If you have the full project repository, you can install all dependencies using the '
              'requirements.txt file:')

    p = anchor_para.insert_paragraph_before()
    run = p.add_run('pip install -r requirements.txt')
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)

    p = anchor_para.insert_paragraph_before()
    p.text = ('Note: The requirements.txt file contains dependencies for the entire project. '
              'For this script specifically, you only need the core dependencies listed above.')

    # 1.4 Utility Modules
    p = anchor_para.insert_paragraph_before()
    p.text = '1.4 Required Utility Modules'
    p.style = 'Heading 2'

    p = anchor_para.insert_paragraph_before()
    p.text = 'The script depends on custom utility modules that must be in your Python path:'

    utils = [
        'utils/sharepoint_utils.py - SharePoint authentication and file operations',
        'utils/azure_blob_utils.py - Azure Blob Storage helper functions',
        'utils/resumable_upload_manager.py - Handles large file uploads',
    ]

    for util in utils:
        p = anchor_para.insert_paragraph_before(util, style='List Bullet')

    p = anchor_para.insert_paragraph_before()
    p.text = ('These modules are included in the project repository and should be in the utils/ directory '
              'relative to the script.')

    # 1.5 Virtual Environment (Recommended)
    p = anchor_para.insert_paragraph_before()
    p.text = '1.5 Using a Virtual Environment (Recommended)'
    p.style = 'Heading 2'

    p = anchor_para.insert_paragraph_before()
    p.text = 'It is recommended to use a Python virtual environment to isolate dependencies:'

    # Step 1
    p = anchor_para.insert_paragraph_before()
    p.add_run('Step 1: Create a virtual environment').bold = True

    p = anchor_para.insert_paragraph_before()
    run = p.add_run('python -m venv venv')
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)

    # Step 2
    p = anchor_para.insert_paragraph_before()
    p.add_run('Step 2: Activate the virtual environment').bold = True

    p = anchor_para.insert_paragraph_before()
    p.text = 'On Linux/macOS:'

    p = anchor_para.insert_paragraph_before()
    run = p.add_run('source venv/bin/activate')
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    p.paragraph_format.left_indent = Inches(0.5)

    p = anchor_para.insert_paragraph_before()
    p.text = 'On Windows:'

    p = anchor_para.insert_paragraph_before()
    run = p.add_run('venv\\Scripts\\activate')
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    p.paragraph_format.left_indent = Inches(0.5)

    # Step 3
    p = anchor_para.insert_paragraph_before()
    p.add_run('Step 3: Install dependencies').bold = True

    p = anchor_para.insert_paragraph_before()
    run = p.add_run('pip install azure-storage-blob PyYAML requests')
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)

    # 1.6 Verifying Installation
    p = anchor_para.insert_paragraph_before()
    p.text = '1.6 Verifying Installation'
    p.style = 'Heading 2'

    p = anchor_para.insert_paragraph_before()
    p.text = 'To verify that all dependencies are installed correctly, run:'

    verify_code = '''python -c "import azure.storage.blob; import yaml; import requests; print('All dependencies installed successfully!')"'''

    p = anchor_para.insert_paragraph_before()
    run = p.add_run(verify_code)
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)

    p = anchor_para.insert_paragraph_before()
    p.text = 'If this command runs without errors, your environment is ready.'

    # 1.7 Configuration Setup
    p = anchor_para.insert_paragraph_before()
    p.text = '1.7 Configuration File Setup'
    p.style = 'Heading 2'

    p = anchor_para.insert_paragraph_before()
    p.text = 'After installing dependencies, you need to configure the script:'

    config_steps = [
        'Create the config directory: mkdir -p config',
        'Copy the template configuration: cp config/tar_transfer_config.yaml.template config/tar_transfer_config.yaml',
        'Edit the configuration file with your Azure and SharePoint credentials',
        'Set up environment variables for sensitive credentials (recommended)',
    ]

    for i, step in enumerate(config_steps, 1):
        p = anchor_para.insert_paragraph_before(f'{i}. {step}', style='List Number')

    # Environment variables example
    p = anchor_para.insert_paragraph_before()
    p.text = 'Example: Setting environment variables for credentials'

    env_example = '''export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;..."
export AZURE_CLIENT_SECRET="your-client-secret-here"'''

    p = anchor_para.insert_paragraph_before()
    run = p.add_run(env_example)
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)

    # Add page break before next section
    p = anchor_para.insert_paragraph_before()
    run = p.add_run()
    run.add_break()

    print("Environment Setup section inserted successfully!")

def update_section_numbers(doc):
    """Update section numbers throughout the document."""
    # Mapping of old to new section numbers
    updates = {
        '1. Overview': '2. Overview',
        '2. Key Features': '3. Key Features',
        '3. Architecture': '4. Architecture & Design',
        '4. Configuration': '5. Configuration',
        '5. Main Components': '6. Main Components',
        '6. Transfer Methods': '7. Transfer Methods',
        '7. Usage Instructions': '8. Usage Instructions',
        '8. Logging': '9. Logging & Monitoring',
        '9. Error Handling': '10. Error Handling',
        '10. Best Practices': '11. Best Practices',
    }

    for para in doc.paragraphs:
        for old, new in updates.items():
            if para.text.strip().startswith(old):
                # Update heading text
                if para.style.name.startswith('Heading'):
                    para.text = para.text.replace(old, new)
                break

def main():
    """Main function to update the documentation."""
    doc_path = '/home/vision_ai_adm/code/oslo/combined_branch/SharePoint_TAR_to_Azure_Blob_Documentation.docx'

    # Load the document
    doc = Document(doc_path)

    # Insert the Environment Setup section
    insert_environment_setup(doc)

    # Update section numbers
    update_section_numbers(doc)

    # Save the updated document
    doc.save(doc_path)
    print(f"Documentation updated successfully: {doc_path}")

if __name__ == "__main__":
    main()
