"""
this file contains the Dataset class for reading the dataset file,
 split it to train and test and get the information of the dataset.
"""
from csv import reader
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import re
import os

class Dataset:
    def __init__(self, dataset_loc):
        # initialize the data set location in txt format
        self.dataset_loc = dataset_loc

        # read the input file in txt format
        self.file_txt = self.read_file()

        # process the dataset by removing the punctuation
        # and split each line into the sentence 1 and sentence 2
        self.dataset = self.txt_processing()

        # get the name of the dataset
        self.name = self.get_dataset_name()
        return
    
    # reads the dataset file
    def read_file(self):
      with open(self.dataset_loc, 'r') as f:
        return (f.readlines())

    # process the txt
    def txt_processing(self):
      # ds stores data as a list that each element is a list with two elements: sentence 1 and sentence 2
      ds = []
      for i in self.file_txt:
        sentences = list(i.split("\t"))[:2]
        for i in range(len(sentences)):
          # remove extra space
          sentences[i].replace("  ", " ")
          # remove punctuation
          sentences[i] = re.sub(r'[^\w\s]', '', sentences[i])
          # convert letters to lowercase
          sentences[i] = sentences[i].lower()
        
        # if the data of this line is not a data record 
        # like newline command at the end of the document
        if len(sentences) !=2:
            continue

        ds.append(sentences)
      return ds

    # extract the name of the dataset from the format that it is written
    def get_dataset_name(self):
      name = ""
      head, tail = os.path.split(self.dataset_loc)
      name = tail.split(".")[-2]
      return name
