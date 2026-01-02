import os
import pandas as pd
import xml.etree.ElementTree as ET
import re

def markdown_to_html(text):
    """
    Convert markdown formatting to HTML-style formatting (str to str)
    Supports:
    - Titles: # text → <h1>text</h1>, ## text → <h2>text</h2>, etc.
    - Bold text: **text** → <strong>text</strong>
    - Italic text: *text* or _text_ → <em>text</em>
    - Underlined text: __text__ → <u>text</u>
    - Unordered lists: * item or - item → <ul><li>item</li></ul>
    - Single line breaks: \n → <br>
    - Paragraphs: double line breaks → <p>text</p> tags
    """
    if not text or not isinstance(text, str):
        return text
    
    # Normalize line breaks
    text = text.replace('\r\n', '\n')
    
    # Convert headers (must be done line by line before paragraph processing)
    # Match headers at the beginning of a line: # Title → <h1>Title</h1>
    text = re.sub(r'^######\s+(.+)$', r'<h6>\1</h6>', text, flags=re.MULTILINE)
    text = re.sub(r'^#####\s+(.+)$', r'<h5>\1</h5>', text, flags=re.MULTILINE)
    text = re.sub(r'^####\s+(.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^###\s+(.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^#\s+(.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    
    # Convert bold text: **text** → <strong>text</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Convert underlined text: __text__ → <u>text</u> (must be done before italic)
    text = re.sub(r'__(.+?)__', r'<u>\1</u>', text)
    # Convert italic text: *text* or _text_ → <em>text</em>
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
    
    # Convert unordered lists: detect lines starting with * or - and group them
    lines = text.split('\n')
    processed_lines = []
    in_list = False
    for line in lines:
        # Check if line is a list item (starts with * or - followed by space)
        if re.match(r'^[\*\-]\s+(.+)$', line):
            list_content = re.match(r'^[\*\-]\s+(.+)$', line).group(1)
            if not in_list:
                processed_lines.append('<ul>')
                in_list = True
            processed_lines.append(f'<li>{list_content}</li>')
        else:
            if in_list:
                processed_lines.append('</ul>')
                in_list = False
            processed_lines.append(line)
    # Close list if we ended while in one
    if in_list:
        processed_lines.append('</ul>')
    text = '\n'.join(processed_lines)
    
    # Convert single line breaks to <br>, but not around block-level elements
    # Split into lines and process each, checking context
    lines = text.split('\n')
    result_lines = []
    block_tags = ['<ul>', '</ul>', '<ol>', '</ol>', '<li>', '</li>', 
                  '<h1>', '</h1>', '<h2>', '</h2>', '<h3>', '</h3>', 
                  '<h4>', '</h4>', '<h5>', '</h5>', '<h6>', '</h6>', '<p>', '</p>']
    
    for i, line in enumerate(lines):
        result_lines.append(line)
        # Check if we should add <br> after this line
        if i < len(lines) - 1:  # Not the last line
            # Don't add <br> if current line ends with a block tag
            ends_with_block = any(line.rstrip().endswith(tag) for tag in block_tags)
            # Don't add <br> if next line starts with a block tag
            next_starts_with_block = any(lines[i + 1].lstrip().startswith(tag) for tag in block_tags)
            # Don't add <br> if next line is empty (paragraph break)
            next_is_empty = lines[i + 1].strip() == ''
            
            if not (ends_with_block or next_starts_with_block or next_is_empty):
                result_lines.append('<br>')
    
    text = '\n'.join(result_lines)
    
    # Convert paragraphs: split by double line breaks and wrap each paragraph in <p> tags
    # Split by double line breaks (paragraph separators)
    paragraphs = text.split('\n\n')
    # Wrap each paragraph in <p> tags
    paragraphs = [f'<p>{p.strip()}</p>' for p in paragraphs if p.strip()]
    # Join paragraphs
    text = '\n'.join(paragraphs)
    
    return text

def get_files():
    prefix = "Testbatterie_FRAG_Rel&Val_with_LLM_response_"
    suffix = ".xlsx"
    
    files = []
    
    # Get all files in current directory
    for filename in os.listdir('.'):
        # Check if it matches our pattern
        if filename.startswith(prefix) and filename.endswith(suffix):
            # Extract the model name between prefix and suffix
            start_idx = len(prefix)
            end_idx = filename.rfind(suffix)
            model_name = filename[start_idx:end_idx]
            
            files.append((filename, model_name))
    
    return files

def remove_thinking(text):
    if "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        return text[:start] + text[end:]
    return text

def load_files():
    files = get_files()
    data_frames = {}
    for filename, model_name in files:
        df = pd.read_excel(filename)
        df['frag_api_response'] = df['frag_api_response'].apply(remove_thinking)
        df['frag_api_response'] = df['frag_api_response'].apply(markdown_to_html)
        data_frames[model_name] = df
    return data_frames

def read_example_LSS(filename="limesurvey_survey_example.lss"):
    """Read and parse the example LSS file."""
    tree = ET.parse(filename)
    root = tree.getroot()
    return tree, root

def create_LSS_template(template_file="limesurvey_survey_example.lss", output_file='survey.lss'):
    """
    Create an LSS file using the template structure.
    Keeps everything from the template except groups, questions, subquestions, answers, and their localizations.
    These sections are prepared as empty placeholders for you to fill in.
    """
    # Parse the template
    tree, root = read_example_LSS(template_file)
    
    # Change language from 'fr' to 'de'
    languages_elem = root.find('languages/language')
    if languages_elem is not None:
        languages_elem.text = 'de'
    
    # Update survey language settings
    survey_lang_elem = root.find('.//surveys_languagesettings/rows/row/surveyls_language')
    if survey_lang_elem is not None:
        survey_lang_elem.text = 'de'
    
    # Update survey metadata language
    survey_language_elem = root.find('.//surveys/rows/row/language')
    if survey_language_elem is not None:
        survey_language_elem.text = 'de'
    
    # Remove the sections we'll replace with our own data
    sections_to_clear = [
        'answers',
        'answer_l10ns', 
        'groups',
        'group_l10ns',
        'questions',
        'subquestions',
        'question_l10ns',
        'question_attributes'
    ]
    
    for section in sections_to_clear:
        element = root.find(section)
        if element is not None:
            root.remove(element)
    
    # Create empty placeholder sections with proper structure
    # You'll fill these in later with your actual data
    
    # Answers section
    answers = ET.SubElement(root, 'answers')
    answers_fields = ET.SubElement(answers, 'fields')
    for field in ['aid', 'qid', 'code', 'sortorder', 'assessment_value', 'scale_id']:
        ET.SubElement(answers_fields, 'fieldname').text = field
    ET.SubElement(answers, 'rows')  # Empty rows - you'll add your answers here
    
    # Answer localizations section
    answer_l10ns = ET.SubElement(root, 'answer_l10ns')
    answer_l10ns_fields = ET.SubElement(answer_l10ns, 'fields')
    for field in ['id', 'aid', 'answer', 'language']:
        ET.SubElement(answer_l10ns_fields, 'fieldname').text = field
    ET.SubElement(answer_l10ns, 'rows')  # Empty rows
    
    # Groups section
    groups = ET.SubElement(root, 'groups')
    groups_fields = ET.SubElement(groups, 'fields')
    for field in ['gid', 'sid', 'group_order', 'randomization_group', 'grelevance']:
        ET.SubElement(groups_fields, 'fieldname').text = field
    ET.SubElement(groups, 'rows')  # Empty rows - you'll add your groups here
    
    # Group localizations section
    group_l10ns = ET.SubElement(root, 'group_l10ns')
    group_l10ns_fields = ET.SubElement(group_l10ns, 'fields')
    for field in ['id', 'gid', 'group_name', 'description', 'language', 'sid', 
                  'group_order', 'randomization_group', 'grelevance']:
        ET.SubElement(group_l10ns_fields, 'fieldname').text = field
    ET.SubElement(group_l10ns, 'rows')  # Empty rows
    
    # Questions section
    questions = ET.SubElement(root, 'questions')
    questions_fields = ET.SubElement(questions, 'fields')
    for field in ['qid', 'parent_qid', 'sid', 'gid', 'type', 'title', 'preg', 
                  'other', 'mandatory', 'encrypted', 'question_order', 'scale_id',
                  'same_default', 'relevance', 'question_theme_name', 'modulename', 'same_script']:
        ET.SubElement(questions_fields, 'fieldname').text = field
    ET.SubElement(questions, 'rows')  # Empty rows - you'll add your questions here
    
    # Subquestions section
    subquestions = ET.SubElement(root, 'subquestions')
    subquestions_fields = ET.SubElement(subquestions, 'fields')
    for field in ['qid', 'parent_qid', 'sid', 'gid', 'type', 'title', 'preg',
                  'other', 'mandatory', 'encrypted', 'question_order', 'scale_id',
                  'same_default', 'relevance', 'question_theme_name', 'modulename', 'same_script']:
        ET.SubElement(subquestions_fields, 'fieldname').text = field
    ET.SubElement(subquestions, 'rows')  # Empty rows - you'll add your subquestions here
    
    # Question localizations section
    question_l10ns = ET.SubElement(root, 'question_l10ns')
    question_l10ns_fields = ET.SubElement(question_l10ns, 'fields')
    for field in ['id', 'qid', 'question', 'help', 'script', 'language']:
        ET.SubElement(question_l10ns_fields, 'fieldname').text = field
    ET.SubElement(question_l10ns, 'rows')  # Empty rows - you'll add your localizations here
    
    # Question attributes section
    question_attributes = ET.SubElement(root, 'question_attributes')
    question_attributes_fields = ET.SubElement(question_attributes, 'fields')
    for field in ['qid', 'attribute', 'value', 'language']:
        ET.SubElement(question_attributes_fields, 'fieldname').text = field
    ET.SubElement(question_attributes, 'rows')  # Empty rows - you'll add attributes if needed
    
    # Ensure sections are in the correct order
    desired_order = [
        'LimeSurveyDocType', 'DBVersion', 'languages',
        'answers', 'answer_l10ns',
        'groups', 'group_l10ns',
        'questions', 'subquestions', 'question_l10ns', 'question_attributes',
        'surveys', 'surveys_languagesettings', 'themes', 'themes_inherited'
    ]
    
    # Reorder elements
    ordered_elements = []
    for tag in desired_order:
        elem = root.find(tag)
        if elem is not None:
            ordered_elements.append(elem)
            root.remove(elem)
    
    # Add any remaining elements that weren't in our list
    for elem in list(root):
        ordered_elements.append(elem)
        root.remove(elem)
    
    # Re-add in correct order
    for elem in ordered_elements:
        root.append(elem)
    
    # Write the file
    ET.indent(tree, space=' ')
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"LSS template created at {output_file}")
    print("Ready for you to add:")
    print("  - Answers in <answers><rows>")
    print("  - Answer localizations in <answer_l10n><rows>")
    print("  - Groups in <groups><rows>")
    print("  - Group localizations in <group_l10n><rows>")
    print("  - Questions in <questions><rows>")
    print("  - Subquestions in <subquestions><rows>")
    print("  - Question localizations in <question_l10n><rows>")
    
    return tree, root

def populate_survey(tree, root, data_frames, survey_id=None):
    """
    Populate the LSS template with actual survey data.
    
    Args:
        tree: ET tree from create_LSS_template
        root: ET root element from create_LSS_template
        data_frames: dict of {model_name: DataFrame}
        survey_id: Survey ID to use (if None, extracts from template)
    """
    if survey_id is None:
        # Extract survey_id from the template
        survey_elem = root.find('.//surveys/rows/row/sid')
        survey_id = int(survey_elem.text) if survey_elem is not None else 1
    
    # Get the first dataframe to extract question structure
    first_df = next(iter(data_frames.values()))
    model_names = list(data_frames.keys())
    
    # Create groups (Handlungsfeld + Aspekt combinations)
    group_map = populate_groups(root, first_df, survey_id)
    
    # Create questions and subquestions
    populate_questions_and_subquestions(root, first_df, model_names, data_frames, 
                                       group_map, survey_id)
    
    return tree, root

def populate_groups(root, df, survey_id):
    """
    Create groups from Handlungsfeld + Aspekt combinations.
    Returns a mapping of (handlungsfeld, aspekt) -> gid
    """
    groups_rows = root.find('.//groups/rows')
    group_l10ns_rows = root.find('.//group_l10ns/rows')
    
    # Get unique combinations of Handlungsfeld and Aspekt, maintaining order
    seen = set()
    unique_combinations = []
    for _, row in df.iterrows():
        combo = (row['Handlungsfeld'], row['Aspekt'])
        if combo not in seen:
            seen.add(combo)
            unique_combinations.append(combo)
    
    group_map = {}
    
    for idx, (handlungsfeld, aspekt) in enumerate(unique_combinations, 1):
        gid = idx
        group_name = f"{handlungsfeld} - {aspekt}"
        
        # Create group structure (language-independent)
        group_row = ET.SubElement(groups_rows, 'row')
        ET.SubElement(group_row, 'gid').text = str(gid)
        ET.SubElement(group_row, 'sid').text = str(survey_id)
        ET.SubElement(group_row, 'group_order').text = str(idx)
        ET.SubElement(group_row, 'randomization_group')
        ET.SubElement(group_row, 'grelevance').text = '1'
        
        # Create group localization (language-dependent text) - now in German
        group_l10n_row = ET.SubElement(group_l10ns_rows, 'row')
        ET.SubElement(group_l10n_row, 'id').text = str(gid)
        ET.SubElement(group_l10n_row, 'gid').text = str(gid)
        ET.SubElement(group_l10n_row, 'group_name').text = group_name
        ET.SubElement(group_l10n_row, 'description')
        ET.SubElement(group_l10n_row, 'language').text = 'de'
        ET.SubElement(group_l10n_row, 'sid').text = str(survey_id)
        ET.SubElement(group_l10n_row, 'group_order').text = str(idx)
        ET.SubElement(group_l10n_row, 'randomization_group')
        ET.SubElement(group_l10n_row, 'grelevance').text = '1'
        
        group_map[(handlungsfeld, aspekt)] = gid
    
    return group_map

def populate_questions_and_subquestions(root, df, model_names, data_frames, 
                                        group_map, survey_id):
    """
    Create questions (test questions) and subquestions (LLM models).
    Each question is an array type where users rate each model's response.
    """
    questions_rows = root.find('.//questions/rows')
    subquestions_rows = root.find('.//subquestions/rows')
    question_l10ns_rows = root.find('.//question_l10ns/rows')
    question_attributes_rows = root.find('.//question_attributes/rows')
    answers_rows = root.find('.//answers/rows')
    answer_l10ns_rows = root.find('.//answer_l10ns/rows')
    
    qid_counter = 1
    l10n_id_counter = 1
    aid_counter = 1
    
    # Answer labels in German - just numbers (1 to 6)
    answer_labels = ["1", "2", "3", "4", "5", "6"]
    
    # Scale reminder to show after each response
    scale_reminder = """
<div style='background-color:#f0f8ff; border-left:4px solid #0066cc; padding:10px; margin:10px 0;'>
<p><strong>Hinweis:</strong> Bitte bewerten Sie die Qualität der Antwort auf einer Skala von 1 bis 6.</p>
<ul style='list-style-type:none; padding-left:0;'>
<li><strong>1</strong> - Unzureichend</li>
<li><strong>2</strong> - Mangelhaft</li>
<li><strong>3</strong> - Befriedigend</li>
<li><strong>4</strong> - Gut</li>
<li><strong>5</strong> - Sehr gut</li>
<li><strong>6</strong> - Ausgezeichnet</li>
</ul>
</div>
"""
    
    for q_order, (idx, row) in enumerate(df.iterrows(), 1):
        frage_id = row['Frage_ID']
        frage = row['Frage']
        handlungsfeld = row['Handlungsfeld']
        aspekt = row['Aspekt']
        komplexitaet = row['Komplexitätsstufe']
        
        gid = group_map[(handlungsfeld, aspekt)]
        main_qid = qid_counter
        qid_counter += 1
        
        # Create main question (array type 'F')
        question_row = ET.SubElement(questions_rows, 'row')
        ET.SubElement(question_row, 'qid').text = str(main_qid)
        ET.SubElement(question_row, 'parent_qid').text = '0'
        ET.SubElement(question_row, 'sid').text = str(survey_id)
        ET.SubElement(question_row, 'gid').text = str(gid)
        ET.SubElement(question_row, 'type').text = 'F'  # Array (Numbers) type
        ET.SubElement(question_row, 'title').text = frage_id
        ET.SubElement(question_row, 'preg')
        ET.SubElement(question_row, 'other').text = 'N'
        ET.SubElement(question_row, 'mandatory').text = 'Y'  
        ET.SubElement(question_row, 'encrypted').text = 'N'
        ET.SubElement(question_row, 'question_order').text = str(q_order)
        ET.SubElement(question_row, 'scale_id').text = '0'
        ET.SubElement(question_row, 'same_default').text = '0'
        ET.SubElement(question_row, 'relevance').text = '1'
        ET.SubElement(question_row, 'question_theme_name').text = 'arrays/array'
        ET.SubElement(question_row, 'modulename')
        ET.SubElement(question_row, 'same_script').text = '0'
        
        # Create answers for this question (1-6 scale)
        for i in range(1, 7):
            answer_row = ET.SubElement(answers_rows, 'row')
            ET.SubElement(answer_row, 'aid').text = str(aid_counter)
            ET.SubElement(answer_row, 'qid').text = str(main_qid)  # Link to this question
            ET.SubElement(answer_row, 'code').text = str(i)  # Order: 1,2,3,4,5,6
            ET.SubElement(answer_row, 'sortorder').text = str(i - 1)
            ET.SubElement(answer_row, 'assessment_value').text = '0'
            ET.SubElement(answer_row, 'scale_id').text = '0'
            
            # Create answer localization - just numbers
            answer_l10n_row = ET.SubElement(answer_l10ns_rows, 'row')
            ET.SubElement(answer_l10n_row, 'id').text = str(aid_counter)
            ET.SubElement(answer_l10n_row, 'aid').text = str(aid_counter)
            ET.SubElement(answer_l10n_row, 'answer').text = answer_labels[i - 1]
            ET.SubElement(answer_l10n_row, 'language').text = 'de'
            
            aid_counter += 1
        
        # Create question localization
        question_l10n_row = ET.SubElement(question_l10ns_rows, 'row')
        ET.SubElement(question_l10n_row, 'id').text = str(l10n_id_counter)
        l10n_id_counter += 1
        ET.SubElement(question_l10n_row, 'qid').text = str(main_qid)
        
        # Format question with just metadata and question text
        full_question = f"<p><strong>{frage_id}</strong> ({komplexitaet})</p>\n<p>{frage}</p>"
        ET.SubElement(question_l10n_row, 'question').text = full_question
        ET.SubElement(question_l10n_row, 'help')
        ET.SubElement(question_l10n_row, 'script')
        ET.SubElement(question_l10n_row, 'language').text = 'de'
        
        # Create subquestions (one per model - these are the rows in the array)
        for model_idx, model_name in enumerate(model_names):
            sub_qid = qid_counter
            qid_counter += 1
            
            # Get the response for this model and question
            model_df = data_frames[model_name]
            response_row = model_df[model_df['Frage_ID'] == frage_id].iloc[0]
            response_text = response_row['frag_api_response']
            
            # Create subquestion structure
            subquestion_row = ET.SubElement(subquestions_rows, 'row')
            ET.SubElement(subquestion_row, 'qid').text = str(sub_qid)
            ET.SubElement(subquestion_row, 'parent_qid').text = str(main_qid)
            ET.SubElement(subquestion_row, 'sid').text = str(survey_id)
            ET.SubElement(subquestion_row, 'gid').text = str(gid)
            ET.SubElement(subquestion_row, 'type').text = 'T'
            ET.SubElement(subquestion_row, 'title').text = f'SQ{model_idx + 1:03d}'
            ET.SubElement(subquestion_row, 'preg')
            ET.SubElement(subquestion_row, 'other').text = 'N'
            ET.SubElement(subquestion_row, 'mandatory')  # Empty for subquestions
            ET.SubElement(subquestion_row, 'encrypted').text = 'N'
            ET.SubElement(subquestion_row, 'question_order').text = str(model_idx + 1)
            ET.SubElement(subquestion_row, 'scale_id').text = '0'
            ET.SubElement(subquestion_row, 'same_default').text = '0'
            ET.SubElement(subquestion_row, 'relevance').text = '1'
            ET.SubElement(subquestion_row, 'question_theme_name')
            ET.SubElement(subquestion_row, 'modulename')
            ET.SubElement(subquestion_row, 'same_script').text = '0'
            
            # Create subquestion localization with response text AND scale reminder
            subquestion_l10n_row = ET.SubElement(question_l10ns_rows, 'row')
            ET.SubElement(subquestion_l10n_row, 'id').text = str(l10n_id_counter)
            l10n_id_counter += 1
            ET.SubElement(subquestion_l10n_row, 'qid').text = str(sub_qid)
            # Display response text followed by scale reminder
            subquestion_with_reminder = f"{response_text}\n{scale_reminder}"
            ET.SubElement(subquestion_l10n_row, 'question').text = subquestion_with_reminder
            ET.SubElement(subquestion_l10n_row, 'help')
            ET.SubElement(subquestion_l10n_row, 'script')
            ET.SubElement(subquestion_l10n_row, 'language').text = 'de'
        
        # Add question attributes
        attr_row1 = ET.SubElement(question_attributes_rows, 'row')
        ET.SubElement(attr_row1, 'qid').text = str(main_qid)
        ET.SubElement(attr_row1, 'attribute').text = 'array_filter'
        ET.SubElement(attr_row1, 'value')
        ET.SubElement(attr_row1, 'language')
        
        # Hide "no answer" option
        attr_row2 = ET.SubElement(question_attributes_rows, 'row')
        ET.SubElement(attr_row2, 'qid').text = str(main_qid)
        ET.SubElement(attr_row2, 'attribute').text = 'hide_tip'
        ET.SubElement(attr_row2, 'value').text = '1'
        ET.SubElement(attr_row2, 'language')
        
        # Random order for models
        attr_row3 = ET.SubElement(question_attributes_rows, 'row')
        ET.SubElement(attr_row3, 'qid').text = str(main_qid)
        ET.SubElement(attr_row3, 'attribute').text = 'random_order'
        ET.SubElement(attr_row3, 'value').text = '1'
        ET.SubElement(attr_row3, 'language')
        
        # Set answer width to 80% to make question column wider
        attr_row4 = ET.SubElement(question_attributes_rows, 'row')
        ET.SubElement(attr_row4, 'qid').text = str(main_qid)
        ET.SubElement(attr_row4, 'attribute').text = 'answer_width'
        ET.SubElement(attr_row4, 'value').text = '80'
        ET.SubElement(attr_row4, 'language')

def create_complete_survey(template_file="limesurvey_survey_example.lss", 
                          output_file='llm_evaluation_survey.lss'):
    """
    Complete workflow: load data, create template, populate, and save.
    """
    # Load the data
    print("Loading data files...")
    data_frames = load_files()
    print(f"Loaded {len(data_frames)} model response files")
    
    # Create template
    print("Creating LSS template...")
    tree, root = create_LSS_template(template_file, 'temp_survey.lss')
    
    # Populate with data
    print("Populating survey with data...")
    tree, root = populate_survey(tree, root, data_frames)
    
    # Save final file
    print("Saving final survey...")
    ET.indent(tree, space=' ')
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    print(f"\n✓ Survey created successfully: {output_file}")
    print(f"  - Groups: {len(root.findall('.//groups/rows/row'))}")
    print(f"  - Questions: {len(root.findall('.//questions/rows/row'))}")
    print(f"  - Subquestions: {len(root.findall('.//subquestions/rows/row'))}")
    print(f"  - Answers: {len(root.findall('.//answers/rows/row'))}")
    
    return tree, root

# Usage
if __name__ == "__main__":
    tree, root = create_complete_survey()