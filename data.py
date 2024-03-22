# -*- coding: utf-8 -*-
"""
Helper file to hide the mess of getting data from various sources
"""
import os





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
        return train_file, test_file, None
    elif data_type.lower() == "max_planck":
        data_dir = '/research_storage/Data_MaxPlanckInstitut/'
        # in Data_MaxPlanckInstitut/output, there are folders named as seg87, seg86b, seg01
        # and a maxplanckdata folder which contains all the txt files (duplicate of the seg* folders content)
        # actual data is in the seg*/input/*.txt
        # command to display size (in lines) of each subdirectory (it takes quite long to execute):
        #       for dir in ./seg*/; do echo $dir $(ls $dir/input/*.txt | xargs cat | wc -l); done
        train_folders = ["seg01","seg02","seg03","seg04","seg05","seg06","seg07","seg08","seg09","seg10","seg11a","seg11b","seg12","seg13","seg14","seg15","seg16","seg17","seg18","seg19","seg20","seg21","seg22","seg23","seg24","seg25","seg26","seg27","seg28","seg29","seg30","seg31","seg32","seg33","seg34","seg35","seg36","seg37","seg38","seg39","seg40","seg41","seg42","seg43","seg44","seg45","seg46","seg47","seg48","seg49","seg50","seg51","seg52","seg53","seg54","seg55","seg56","seg57","seg58","seg59","seg60","seg61","seg62","seg63","seg64","seg65","seg66","seg67","seg68","seg69","seg70",]
        test_folders = ["seg71","seg72","seg73","seg74","seg75","seg76","seg77","seg78","seg79",]
        val_folders = ["seg80","seg81","seg82","seg83","seg84","seg85","seg86a","seg86b","seg87"]
        # replace the "segXX" with a path to txt files, using a * wildcard
        for folders_list in [train_folders, test_folders, val_folders]:
            folders_list = [f"{data_dir}output/{seg}/input/{seg}*.txt" for seg in folders_list]

        return tuple(folders_list)




 
# dataset = load_dataset("text", data_files={"train":['/research_storage/Data_MaxPlanckInstitut/output/seg01/input/seg01*.txt','/research_storage/Data_MaxPlanckInstitut/output/seg02/input/seg02*.txt'], "test":'/research_storage/Data_MaxPlanckInstitut/output/seg03/input/seg03*.txt'})
