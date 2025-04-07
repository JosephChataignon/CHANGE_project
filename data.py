# -*- coding: utf-8 -*-
"""
Helper file to hide the mess of getting data from various sources
"""
import os, logging, random, unicodedata
from datasets import load_dataset, Dataset, DatasetDict
from collections import defaultdict
from nltk.tokenize.punkt import PunktSentenceTokenizer


def get_file_paths(root_dir: str, file_extensions: list[str]) -> list[str]:
    """
    Retrieves a list of paths to all files with specified extensions in the given 
    root directory and its subdirectories.
    """
    file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(filename.endswith(f".{ext}") for ext in file_extensions):
                file_paths.append(os.path.join(dirpath, filename))
    
    return file_paths


def get_CHANGE_data(data_type, data_storage):
    assert isinstance(data_storage, str) and len(data_storage) > 0, f"data_storage should be a non-empty string, got '{data_storage}'"

    if data_type.lower() == "walser":
        data_dir = os.path.join(data_storage,'Projekt_Change_LLM/Walser_data/')
        train_file = data_dir + "train_dataset.txt"
        test_file  = data_dir + "test_dataset.txt"
        return load_dataset("text", data_files={"train":train_file, "test":test_file})
    
    elif "maxplanck" in data_type.lower().replace('-',''):
        data_dir = os.path.join(data_storage,'Data_MaxPlanckInstitut/')
        substitutions_file = data_dir + 'scripts/CHANGE_processing/unique_characters-replace.txt'
        # We will need to substitute some problematic characters with new ones
        character_pairs = load_substitutions(substitutions_file)
        def substitute_chars(line):
            # do the substitution
            for k,v in character_pairs.items():
                if v == "DEL": v=""
                line['text'] = line['text'].replace(k,v)
            # convert string to Unicode's Normal Form C (NFC), grouping diacritics with letters when possible
            line['text'] = unicodedata.normalize('NFC', line['text'])
            # removing isolated diacritics (they should be the less common ones)
            #line['text'] = ''.join(c for c in line['text'] if unicodedata.category(c) not in ['Mn','Ni','No','Lm','Lo'])
            line['text'] = ''.join(c for c in line['text'] if ord(c)<=128)
            return line
        # in Data_MaxPlanckInstitut/output, there are folders named as seg87, seg86b, seg01
        # and a maxplanckdata folder which contains all the txt files (duplicate of the seg* folders content)
        # actual data is in the seg*/input/*.txt
        # command to display size (in lines) of each subdirectory (it takes quite long to execute):
        #       for dir in ./seg*/; do echo $dir $(ls $dir/input/*.txt | xargs cat | wc -l); done
        if 'test' in data_type.lower():
            # if it's for testing, just use one file
            dataset = load_dataset("text", 
                    data_files={"train":f"{data_dir}output/seg01/input/seg01_1524_00000030.txt", 
                                "test":f"{data_dir}output/seg71/input/seg71_206422_00000318.txt",
                                "validation":f"{data_dir}output/seg80/input/seg80_231690_00000395.txt" })
        else:
            data_seg = { 'train' : ["seg01","seg02","seg03","seg04","seg05","seg06","seg07","seg08","seg09","seg10",
                                    "seg11a","seg11b","seg12","seg13","seg14","seg15","seg16","seg17","seg18","seg19","seg20",
                                    "seg21","seg22","seg23","seg24","seg25","seg26","seg27","seg28","seg29","seg30",
                                    "seg31","seg32","seg33","seg34","seg35","seg36","seg37","seg38","seg39","seg40",
                                    "seg41","seg42","seg43","seg44","seg45","seg46","seg47","seg48","seg49","seg50",
                                    "seg51","seg52","seg53","seg54","seg55","seg56","seg57","seg58","seg59","seg60",
                                    "seg61","seg62","seg63","seg64","seg65","seg66","seg67","seg68","seg69","seg70",],
                        'test' :   ["seg71","seg72","seg73","seg74","seg75","seg76","seg77","seg78","seg79",],
                        'validation'  :   ["seg80","seg81","seg82","seg83","seg84","seg85","seg86a","seg86b","seg87",]}
            # replace the "segXX" with a path to txt files, using a * wildcard
            data_files = defaultdict(list)
            for type_ in data_seg.keys():
                for seg in data_seg[type_]:
                    data_files[type_].append(f"{data_dir}output/{seg}/input/{seg}*.txt")

            assert all(key in data_files for key in ["train", "test", "validation"]), f"Elements missing from the data files, only {data_files.keys()} are there."
            dataset = load_dataset("text", data_files=data_files)
        logging.info(f"Data selection finished, now applying characters substitution")
        dataset = dataset.map(substitute_chars)
        return dataset
        


def load_substitutions(substitutions_file):
    """ This loads a file of substitutions where each line has 2 characters separated by a tab """
    logging.info(f'preparing to substitute characters in the dataset, from file: {substitutions_file}')
    # always replace superscript small E with umlaut/trema
    character_pairs = {u'\u0308':u'\u0364'}
    # Read the file and extract character pairs
    with open(substitutions_file, 'r', encoding='utf-8') as f:
        for line in f:
            pair = line.strip().split('\t')
            if len(pair) == 2:
                character_pairs[pair[0]] = pair[1]
            else:
                logging.info(f"substitutions: this line does not have exactly 2 elements: {line.strip()}")
    return character_pairs


def get_CHANGE_data_for_sentences(data_type, data_storage):
    assert isinstance(data_storage, str) and len(data_storage) > 0, f"data_storage should be a non-empty string, got '{data_storage}'"

    if data_type.lower() == 'education':
        # find paths of all files
        data_dir = os.path.join(data_storage, 'Projekt_Change_LLM/Eduscience_data')
        # make test/train split - do we actually need it ?
        train_files,test_files = '',''
        # clean the text ? manage footnotes ?
        
        # chunk here?
        
        # load and return ?
        return load_dataset("text", data_files={"train":train_files, "test":test_files})

    elif data_type.lower() == 'education_sample':
        # find paths of all files
        data_dir = os.path.join(data_storage, 'Projekt_Change_LLM/Preprocessed_Eduscience_data/sample_clean')
        data_files = get_file_paths(data_dir,['txt'])
        logging.info(f'Loading dataset education_sample, searching from root:{data_dir}, found {len(data_files)} txt files')
        texts = []
        for file_path in data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                texts.append({"text": text, "file_name": os.path.basename(file_path)})

        # Segment documents into sentences
        dataset = Dataset.from_dict(texts)
        sentence_dataset = dataset.map(segment_documents, batched=True, remove_columns=dataset.column_names)

        # Create triplets
        triplets = create_triplets(sentence_dataset)

        # Split into train, dev, and test sets
        train_test_split = triplets.train_test_split(test_size=0.1)
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']

        dev_test_split = test_dataset.train_test_split(test_size=0.5)
        dev_dataset = dev_test_split['train']
        test_dataset = dev_test_split['test']

        return DatasetDict({
            "train": train_dataset,
            "dev": dev_dataset,
            "test": test_dataset
        })
        

def segment_documents(examples):
    all_sentences = []
    doc_ids = []
    tokenizer = PunktSentenceTokenizer()
    for i, text in enumerate(examples["text"]):
        sentences = tokenizer.tokenize(text)
        all_sentences.extend(sentences)
        doc_ids.extend([i] * len(sentences))
    
    return {"sentence": all_sentences, "doc_id": doc_ids}

def create_triplets(sentence_dataset):
    sentences = sentence_dataset["sentence"]
    doc_ids = sentence_dataset["doc_id"]
    triplets = []

    # Strategy: Sentences from the same document are positive pairs
    # Sentences from different documents are negative pairs
    for i in range(len(sentences)):
        anchor = sentences[i]
        anchor_doc_id = doc_ids[i]

        # Find positive example (from the same document)
        pos_indices = [j for j in range(len(sentences)) if doc_ids[j] == anchor_doc_id and j != i]
        if pos_indices:
            pos_idx = random.choice(pos_indices)
            positive = sentences[pos_idx]

            # Find negative example (from a different document)
            neg_indices = [j for j in range(len(sentences)) if doc_ids[j] != anchor_doc_id]
            if neg_indices:
                neg_idx = random.choice(neg_indices)
                negative = sentences[neg_idx]

                triplets.append({
                    "anchor": anchor,
                    "positive": positive,
                    "negative": negative
                })

    return Dataset.from_list(triplets)