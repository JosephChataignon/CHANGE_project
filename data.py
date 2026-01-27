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


def get_CHANGE_data_for_sentences(data_type, data_storage):
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
    raw_dataset = Dataset.from_list([{
            "text": Path(file_path).read_text(encoding='utf-8'), 
            "file_name": Path(file_path).name
        } for file_path in data_files
    ])
    
    # Clean documents here ? manage footnotes ?
    
    # Chunk documents into sentences
    logging.info('Chunking documents into sentences')
    sentence_dataset = raw_dataset.map(
        segment_documents,
        batched=True,
        with_indices=True,
        remove_columns=raw_dataset.column_names,
    )
    logging.info('Dataset chunked, now creating triplets from sentences')
    triplets = create_triplets(sentence_dataset)

    logging.info(f'Triplet dataset created with {len(triplets)} triplets, now splitting into train/dev/test')
    train_split = triplets.train_test_split(test_size=0.2, seed=42)
    dev_test_split = train_split["test"].train_test_split(test_size=0.5, seed=42)
    return DatasetDict({
        "train": train_split["train"],
        "dev": dev_test_split["train"],
        "test": dev_test_split["test"]
    })
        

def segment_documents(examples, indices=None):
    all_sentences = []
    doc_ids = []
    tokenizer = PunktSentenceTokenizer()
    # Prefer source file_name as stable doc_id; fall back to provided indices, then to position
    file_names = examples.get("file_name")
    doc_labels = file_names if file_names is not None else (indices if indices is not None else range(len(examples["text"])))
    for text, doc_label in zip(examples["text"], doc_labels):
        sentences = tokenizer.tokenize(text)
        all_sentences.extend(sentences)
        doc_ids.extend([doc_label] * len(sentences))
    return {"sentence": all_sentences, "doc_id": doc_ids}

def create_triplets(sentence_dataset, sample_scale=0.3, min_per_doc=10):
    ''' sample_scale: to regulate how many sentences per document are included in training, so the trianing
     runs don't take forever.
    min_per_doc: minimum number of sentences to sample per document
    '''
    sentences = sentence_dataset["sentence"]
    doc_ids = sentence_dataset["doc_id"]
    triplets = []

    # Pass 1: build doc -> indices map and decide sampled indices per doc
    doc_indices = defaultdict(list) # format doc_id -> list of sentence indices
    for idx, doc_id in enumerate(doc_ids):
        doc_indices[doc_id].append(idx)

    sampled_by_doc = {}
    available_docs = []  # docs with >=2 sampled sentences (needed for positives)
    total_sentences = len(sentences)
    sampled_sentences = 0
    for doc_id, indices in doc_indices.items():
        # quota of sentences to sample from this document
        quota = min(min_per_doc, math.ceil(sample_scale * math.sqrt(len(indices))))
        if len(indices) <= quota:
            sampled = indices
        else:
            sampled = random.sample(indices, quota)
        sampled_by_doc[doc_id] = sampled
        sampled_sentences += len(sampled)
        if len(sampled) >= 2:
            available_docs.append(doc_id)

    logging.info(f"Sampling for triplets: kept {sampled_sentences} of {total_sentences} sentences (~{100*sampled_sentences/total_sentences:.2f}%).")

    # Build triplets from sampled pool only
    for doc_id in tqdm(available_docs, desc="Building triplets"):
        idxs = sampled_by_doc[doc_id]
        for anchor_idx in idxs:
            # pick positive from same doc, not the anchor
            if len(idxs) < 2:
                continue
            while True:
                pos_idx = random.choice(idxs)
                if pos_idx != anchor_idx:
                    break
            positive = sentences[pos_idx]

            # pick negative from a different doc
            neg_doc = None
            for _ in range(10):
                candidate_doc = random.choice(available_docs)
                if candidate_doc != doc_id:
                    neg_doc = candidate_doc
                    break
            if neg_doc is None:
                continue
            neg_idx = random.choice(sampled_by_doc[neg_doc])

            triplets.append({
                "anchor": sentences[anchor_idx],
                "positive": positive,
                "negative": sentences[neg_idx]
            })

    return Dataset.from_list(triplets)