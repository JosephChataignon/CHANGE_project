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
        return train_file, test_file
    elif data_type.lower() == "max_planck":
        data_dir = '/research_storage/Data_MaxPlanckInstitut/'
        # in Data_MaxPlanckInstitut/output, there are folders named as seg87, seg86b, seg01
        # and a maxplanckdata folder which I don't know the use of
        # actual data is in the seg*/input/*.txt







 
