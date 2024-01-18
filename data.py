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
            working_dir = "/content/drive/MyDrive/Unibe/"
            data_dir = working_dir
        else:
            # from Ubelix container
            working_dir = '/research_storage/'
            data_dir = working_dir + 'Walser_data/'
        train_file = data_dir + "train_dataset.txt"
        test_file  = data_dir + "test_dataset.txt"
        return train_file, test_file






 
