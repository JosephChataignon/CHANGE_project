from dotenv import load_dotenv
import os

from utils import load_test_batterie, query_frag_api, query_anyllm, query_ollama, search_frag_documents

# Load environment variables from .env file
env_file = '../.env' 
load_dotenv(env_file)
# Check if OLLAMA_HOST is loaded in the environment
assert os.getenv('OLLAMA_HOST') is not None, "OLLAMA_HOST is not set in the environment"

prompt_template = """
DOCUMENTS:

{data}


QUESTION:
{query}


INSTRUCTIONS:
Answer the users QUESTION using the DOCUMENTS text above.
Keep your answer ground in the facts of the DOCUMENT.
If the DOCUMENT doesn't contain the facts to answer the QUESTION return NO Answer found.
"""

def validate_questions(df, responder='frag_api', use_docs=0):
    ''' Reads the battery of test questions and submit them to the selected responder,
    then adds the responses to a new column in the dataframe.
    Args:
        df (pd.DataFrame): DataFrame containing the test questions.
        responder (str): The responder to use ('frag_api', 'deepseekR1', ...).
        use_docs (int): Number of documents to retrieve from FRAG for context. If set to 0, skips document retrieval.
    Returns:
        pd.DataFrame: DataFrame with an additional column for responses.
    '''
    available_responders = ['frag_api', 'gpt5', 'deepseekR1'] 
    assert responder in available_responders, f"Unknown  responder {responder}"
    
    newcolumn = f'{responder}_response'
    # Initialize the new column with None values
    df[newcolumn] = None
    # if use_docs:
    #     df['context_documents'] = None  # Optional: store retrieved documents
    
    for index, row in df.iterrows():
        print(f"Processing question {index}/{len(df)}: ID {row['Frage_ID']}")
        frage = row['Frage']

        # Query the API with the question
        if responder == 'frag_api':
            api_response = query_frag_api(frage, n_results=use_docs)
            response_text = api_response.get('response', str(api_response))
        else:
            if use_docs:
                documents = search_frag_documents(frage, n_results=use_docs)  
                augmented_prompt = prompt_template.format(data=documents['formatted_data'], query=frage) 
                frage = augmented_prompt
            
            api_response = query_anyllm(frage, responder) 
            response_text = api_response.get('response', str(api_response))
    
        # Store the response in the new column
        df.at[index, newcolumn] = response_text
        print(f"Response: {response_text}\n")
    return df


# Run the validation
df = load_test_batterie()
print(f"Loaded {len(df)} questions for validation.")
# In the end we use all with FRAG. Model has to be changed in FRAG config, not through API
for responder in ['frag_api']: # can use ['frag_api', 'gpt5', 'deepseekR1']
    print(f"Validating questions using responder: {responder}")
    df = validate_questions(df,responder=responder,use_docs=12)  # update df with new column

# Save the updated DataFrame to a new Excel file
output_file = 'Testbatterie_FRAG_Rel&Val_with_LLM_response.xlsx'
df.to_excel(output_file, index=False)


