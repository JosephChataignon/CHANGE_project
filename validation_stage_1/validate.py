from dotenv import load_dotenv
import os

from utils import load_test_batterie, query_frag_api, query_anyllm, query_ollama

# Load environment variables from .env file
env_file = '../.env' 
load_dotenv(env_file)
# Check if OLLAMA_HOST is loaded in the environment
assert os.getenv('OLLAMA_HOST') is not None, "OLLAMA_HOST is not set in the environment"


def validate_questions(df, responder='frag_api'):
    ''' Reads the battery of test questions and submit them to the selected responder,
    then adds the responses to a new column in the dataframe.
    Args:
        df (pd.DataFrame): DataFrame containing the test questions.
        responder (str): The responder to use ('frag_api' or 'anythingllm').
    Returns:
        pd.DataFrame: DataFrame with an additional column for responses.
    '''
    available_responders = ['frag_api', 'gpt5', 'deepseek'] #TODO: list all possible responders
    assert responder in available_responders, f"Unknown  responder {responder}"
    
    for index, row in df.iterrows():
        frage = row['Frage']
        
        if responder == 'frag_api':
            newcolumn = 'frag_api_response'
        else:
            newcolumn = 'api_response'

        # Initialize the new column with None values
        df[newcolumn] = None
        
        # Query the API with the question
        if responder == 'frag_api':
            api_response = query_frag_api(frage)
            response_text = api_response.get('response', str(api_response))
        else:
            api_response = query_anyllm(frage, responder) 
            response_text = api_response.get('response', str(api_response))
    
        # Store the response in the new column
        df.at[index, newcolumn] = response_text
    
    return df


# Run the validation
df = load_test_batterie()
for responder in ['frag_api', 'gpt5', 'deepseek']:
    df = validate_questions(df,responder=responder)  # update df with new column

# Save the updated DataFrame to a new Excel file
output_file = 'Testbatterie_FRAG_Rel&Val_with_LLM_response.xlsx'
df.to_excel(output_file, index=False)


