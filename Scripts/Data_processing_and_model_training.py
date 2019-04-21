#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 09:49:58 2019

@author: Hesham El Abd
@Description: The Scripts reads the preprocessed song corpa, encode them using 
numerically, then it prepare the data using tensorflow Data API, then it create
a model and train it using the data. 
"""
import argparse
from utility_functions import(UniqueElements,GenChar2IdxMap,GenIdx2Char,
                              EncodeText,PrepareTrainingTensors)
import pickle
import tensorflow as tf
# parsing user inputs 
parser=argparse.ArgumentParser()

parser.add_argument("-e","--embedding_dimention",
                    help="The embedding dimention of each character",
                    type=int)

parser.add_argument("-n","--num_of_units",
                    help="the number of units used to construct the LSTM units",
                    type=int)

parser.add_argument("-r","--recurrent_dropout",
                    help="the recurrent dropout used to regularize the"+ 
                    "LSTM units",
                    type=float)

parser.add_argument("-i","--input_dropout",
                    help="the dropout applied to the input of the LSTM units",
                    type=float)

parser.add_argument("-b","--batch_size",
                    help="The batch size used to train the model",
                    type=int)

parser.add_argument("-s","--cond_seq_len",
                    help="The length of the conditional string",
                    type=int)

parser.add_argument("-d","--num_epochs",
                    help="nmber of training epochs",
                    type=int)

parser.add_argument("-o","--output",
                    help="The output pass to save the model after training")

user_input=parser.parse_args()
emb_dim=user_input.embedding_dimention
num_units=user_input.num_of_units
recurrent_dropout=user_input.recurrent_dropout
input_dropout=user_input.input_dropout
batch_size=user_input.batch_size
cond_seq_len=user_input.cond_seq_len
num_epochs=user_input.cond_seq_len
output_dir=user_input.output

# Load the proprocessed dataset: 
with open("../Data/song_corpa.txt","r") as infile:
    song_corpa=infile.read()

# prepare the corpa for training: 
unique_elements=UniqueElements(song_corpa)
vocab_size=len(unique_elements)
char2idx=GenChar2IdxMap(unique_elements)
idx2char=GenIdx2Char(unique_elements)

# save a copy of the maps to be used during inferences: 
with open("../Resources/char2idx.pickle","wb") as outfile:
    pickle.dump(char2idx,outfile)

with open("../Resources/idx2char.pickle","wb") as outfile:
    pickle.dump(idx2char,outfile)
    
## Encode the corpa numerically 
Encoded_corpa=EncodeText([song_corpa[0:1e6]],char2idx)











