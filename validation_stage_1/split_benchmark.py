import pandas as pd
from openpyxl import load_workbook

"""
Abandoned script; 
I thought we'd use multiple spreadsheets, one per expert, to gather evaluations,
but in the end we'll use a handmade MS Forms instead.
"""

def split_for_experts(df, experts):
    ''' Splits the dataframe into multiple dataframes, one for each expert.
    Args: 
        df (pd.DataFrame): DataFrame containing the test questions.
        experts (list): list of experts following the format:
            [
                {
                    'name': 'Andre',
                    'questions': {
                        'F01': [A1, A4, ...],
                        'F06': [A3, A6, ...],
                    }
            ...
    '''
    for expert in experts:
        filename = f'Testbatterie_{expert['name']}.xlsx'
        sub_df = df[df['Frage_ID'].isin(expert['questions'].keys())]
        # TODO: filter answers if needed
        # TODO: change names of answer columns
        # TODO: add columns for rating and remarks for each answer
        
        sub_df = sub_df.drop(columns=['Handlungsfeld', 'Komplexitätsstufe', 'Aspekt'], inplace=True)
        # write to file
        sub_df.to_excel(filename, index=False)
        # Load, freeze row 1, save
        wb = load_workbook(filename)
        ws = wb.active
        ws.freeze_panes = "A2"
        wb.save(filename)


def merge_experts_evals(df, experts):
    ''' Merges the files filled by each expert into one file.
    Args: 
        same as the split function
    '''
    for expert in experts:
        filename = f'Testbatterie_{expert['name']}.xlsx'
        expert_df = pd.read_excel(filename)
        # TODO: think of the final structure of the merged file
        # TODO: complete this function


# Read the Excel file
file_to_load = 'Testbatterie_FRAG_Rel&Val_with_LLM_response.xlsx'
df = pd.read_excel(file_to_load)

# TODO: write the logic to assign rows and columns to each expert 
experts = [{'name': 'Andre',
            'questions': {
                'F01': ['A1', 'A4'],
                'F06': ['A3', 'A6'],
            }},
           {'name': 'Joseph',
            'questions': {
                'F02': ['A2', 'A5'],
                'F03': ['A1', 'A4'],
            }}
            ]

split_for_experts(df, experts)
#merge_experts_evals(df, experts)

