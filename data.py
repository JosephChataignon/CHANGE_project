# -*- coding: utf-8 -*-
"""
Helper file to hide the mess of getting data from various sources
"""
import os, logging, random, unicodedata, math
from tqdm import tqdm
from pathlib import Path
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


def get_CHANGE_data_for_sentences(data_type, data_storage, 
                                  segmentation_method={"method":"sentence", "chunk_size":12, "overlap":2}, 
                                  sample_scale=1, min_triplets_per_doc=10):
    """
    Load and process documents for embeddings fine-tuning with sentence-level triplets.
    
    Parameters:
    data_type : Type of dataset to load ('education' or 'education_sample')
    data_storage : Root directory for data storage
    segmentation_method : Method to segment documents, default "sentence"
    chunk_size : int, default=1
    overlap : int, default=0
    sample_scale : Scaling factor for quota calculation. Higher values = more samples per document.
    min_triplets_per_doc : Minimum number of triplets to sample per document.
    
    Returns:
    DatasetDict : Dictionary with 'train', 'dev', 'test' splits containing triplets
    """
    assert isinstance(data_storage, str) and len(data_storage) > 0, f"data_storage should be a non-empty string, got '{data_storage}'"

    if data_type.lower() == 'education':
        # find paths of all files
        extensions = ['txt']
        data_dir = os.path.join(data_storage, 'Projekt_Change_LLM/Eduscience_data')
    elif data_type.lower() == 'education_sample':
        extensions = ['txt']
        data_dir = os.path.join(data_storage, 'Projekt_Change_LLM/Preprocessed_Eduscience_data/sample_clean')
    
    data_files = get_file_paths(data_dir, extensions)
    logging.info(f'Loading dataset {data_type} from root:{data_dir}, found {len(data_files)} txt files')
    
    # Need to add references (author, etc) from spreadsheet
    
    # Clean documents here ? manage footnotes ?
    
    # New pipeline: process documents one by one
    logging.info(f'Processing documents with method={segmentation_method}')
    all_chunks_by_doc = {}  # doc_id -> list of chunks
    sampled_pairs = []  # Collect sampled pairs as we process each document
    total_pairs_generated = 0
    
    for file_path in tqdm(data_files, desc="Processing documents"):
        # Read document
        text = Path(file_path).read_text(encoding='utf-8')
        doc_id = Path(file_path).name
        
        # Segment document into chunks
        chunks = segment_document(text, doc_id, segmentation_method=segmentation_method)
        if len(chunks) < 2:
            logging.warning(f"Only {len(chunks)} chunks in {doc_id}, skipping")
            continue
        # Store chunks for later negative sampling
        all_chunks_by_doc[doc_id] = chunks
        
        possible_pairs = len(chunks) * (len(chunks) - 1) // 2
        quota = max(min_triplets_per_doc, math.ceil(sample_scale * math.sqrt(possible_pairs)))
        # Create and sample positive pairs from this document
        doc_pairs = create_pairs_from_document(chunks, doc_id, quota=quota)
        sampled_pairs.extend(doc_pairs)
        
        # Track statistics
        total_pairs_generated += possible_pairs
    
    logging.info(f'Sampled {len(sampled_pairs)} pairs from {total_pairs_generated} possible pairs ')
    
    # Add negatives to create triplets
    triplets = add_negatives_to_pairs(sampled_pairs, all_chunks_by_doc)

    logging.info(f'Triplet dataset created with {len(triplets)} triplets, now splitting into train/dev/test')
    train_split = triplets.train_test_split(test_size=0.2, seed=42)
    dev_test_split = train_split["test"].train_test_split(test_size=0.5, seed=42)
    logging.info(f"Final dataset sizes: train={len(train_split['train'])}, dev={len(dev_test_split['train'])}, test={len(dev_test_split['test'])}") 
    
    return DatasetDict({
        "train": train_split["train"],
        "dev": dev_test_split["train"],
        "test": dev_test_split["test"]
    })
        

def segment_document(text, doc_id, segmentation_method):
    """
    Segment a single document into chunks (sentences or groups of sentences).
    returns a list of str : The text chunks from this document
    """
    method, chunk_size, overlap = segmentation_method["method"], segmentation_method["chunk_size"], segmentation_method["overlap"]
    # Segment into base units (sentences)
    if method == "sentence":
        tokenizer = PunktSentenceTokenizer()
        sentences = tokenizer.tokenize(text)
    else:
        raise ValueError(f"Unsupported segmentation method: {method}")
    
    # If chunk_size is 1, return sentences as-is
    if chunk_size == 1:
        return sentences
    
    # Group sentences with sliding window
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(sentences)-overlap, step):
        chunk_end = min(i + chunk_size, len(sentences))
        chunk = " ".join(sentences[i:chunk_end])
        if len(chunk) > 0: chunks.append(chunk)    
    return chunks



def create_pairs_from_document(chunks, doc_id, quota):
    """
    Create all possible positive pairs from chunks and immediately sample them.
    Returns:
    list of dict : Sampled pairs with keys 'anchor', 'positive', 'doc_id'
    """
    n = len(chunks)
    total_possible_pairs = n * (n - 1) // 2
        
    all_pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sampled_indices = random.sample(all_pair_indices, quota) if quota < total_possible_pairs else all_pair_indices
    
    sampled_pairs = []
    for i, j in sampled_indices:
        sampled_pairs.append({
            "anchor": chunks[i],
            "positive": chunks[j],
            "doc_id": doc_id
        })
    return sampled_pairs



def add_negatives_to_pairs(sampled_pairs, all_chunks_by_doc):
    """
    Add negative examples to pairs to create triplets.
    
    Parameters:
    sampled_pairs : list of dict
        Pairs with 'anchor', 'positive', 'doc_id' keys
    all_chunks_by_doc : dict
        Mapping from doc_id to list of chunks from that document
    
    Returns:
    Dataset : HuggingFace Dataset with 'anchor', 'positive', 'negative' columns
    """
    triplets = []
    available_docs = list(all_chunks_by_doc.keys())
    
    for pair in tqdm(sampled_pairs, desc="Adding negatives to create triplets"):
        anchor_doc = pair["doc_id"]
        
        # Select a negative from a different document
        neg_doc = None
        for _ in range(10):
            candidate_doc = random.choice(available_docs)
            if candidate_doc != anchor_doc:
                neg_doc = candidate_doc
                break
        if neg_doc is None: continue
        
        # Pick random chunk from negative document
        negative_chunk = random.choice(all_chunks_by_doc[neg_doc])
        
        triplets.append({
            "anchor": pair["anchor"],
            "positive": pair["positive"],
            "negative": negative_chunk
        })
    
    return Dataset.from_list(triplets)

