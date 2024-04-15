# -*- coding: utf-8 -*-
"""
Helper file to hide the mess of getting data from various sources
"""
import os
import logging
import unicodedata
from datasets import load_dataset





def get_CHANGE_data(data_type):

    if data_type.lower() == "walser":
        if os.getenv("COLAB_RELEASE_TAG"):
            # if executed in Colab, this will try to get
            # the files from Google drive
            from google.colab import drive
            drive.mount('/content/drive')
            data_dir = "/content/drive/MyDrive/Unibe/"
        else:
            # from Ubelix container
            data_dir = '/research_storage/Projekt_Change_LLM/Walser_data/'
        train_file = data_dir + "train_dataset.txt"
        test_file  = data_dir + "test_dataset.txt"
        return load_dataset("text", data_files={"train":train_file, "test":test_file})
    
    elif "maxplanck" in data_type.lower().replace('-',''):
        data_dir = '/research_storage/Data_MaxPlanckInstitut/'
        substitutions_file = data_dir + 'scripts/CHANGE_processing/unique_characters-replace.txt'
        # We will need to substitute some problematic characters with new ones
        character_pairs = load_substitutions(substitutions_file)
        def substitute_chars(line):
            # do the substitution
            for k,v in character_pairs.items():
                line['text'] = line['text'].replace(k,v)
            # convert string to Unicode's Normal Form C (NFC), grouping diacritics with letters when possible
            line['text'] = unicodedata.normalize('NFC', line['text'])
            # removing isolated diacritics (they should be the less common ones)
            line['text'] = ''.join(c for c in line['text'] if unicodedata.category(c) != 'Mn')
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
                        'val'  :   ["seg80","seg81","seg82","seg83","seg84","seg85","seg86a","seg86b","seg87",],}
            # replace the "segXX" with a path to txt files, using a * wildcard
            data_files = {'train': [], 'test' : [], 'validation': []}, 
            for type_ in data_seg.keys():
                for seg in data_seg[type_]:
                    data_files[type_].append(f"{data_dir}output/{seg}/input/{seg}*.txt")

            dataset = load_dataset("text", data_files=data_files)
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


