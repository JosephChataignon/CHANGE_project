import os
import pandas as pd
import xml.etree.ElementTree as ET

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
        #if 'api_response' in df.columns:
        df['api_response'] = df['api_response'].apply(remove_thinking)
        data_frames[model_name] = df
    return data_frames

def read_example_LSS(filename="limesurvey_survey_example.lss"):
    with open(filename, 'r') as file:
        content = file.read()
    return content

def create_XML(data_frames, output_file='survey.lss'):
    root = ET.Element('document')
    
    
    
    
    for model_name, df in data_frames.items():
        model_elem = ET.SubElement(root, 'Model', name=model_name)
        
        for _, row in df.iterrows():
            question_elem = ET.SubElement(model_elem, 'Question', id=str(row['Frage_ID']))
            frage_elem = ET.SubElement(question_elem, 'Frage')
            frage_elem.text = str(row['Frage'])
            
            response_elem = ET.SubElement(question_elem, 'Response')
            response_elem.text = str(row[f'{model_name}_response'])
    
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)



data = load_files()
create_XML(data)
