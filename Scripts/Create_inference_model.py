#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 13:26:40 2019
@author: Hesham EL Abd
@Description: This script is used to load a trained model, copy its weights into
a new model with the same architecture and save the model into a user specific 
directory. 
"""
# load and import functions
import argparse
from utility_functions import CreateModels, LoadModel,loss

## USer specific argments: 
parser=argparse.ArgumentParser()

parser.add_argument("--input", 
                    help="The path to the trained model")

parser.add_argument("--embedding_dimention", 
                    help="The embedding dimentionalility for building"+
                    "the embedding layer of the model."+" it must be the same"+
                    "Dimension used to train the source model." ,
                    type=int)

parser.add_argument("--num_vocab",
                    help="Number of worlds in the vocabulary, it must be"+
                    "the same as used during the training",
                    type=int)

parser.add_argument("--num_of_lstm_units", 
                    help="The number of units to be used to create the"+
                    "LSTM",
                    type=int)



parser.add_argument("--output",
                help="The output directory to save the model after construction")

## Parsing user inputs
user_input=parser.parse_args()

input_model=user_input.input
emb_dim=user_input.embedding_dimention
vocab_size=user_input.num_vocab
num_units=user_input.num_of_lstm_units
output_dir=user_input.output

## Load the model
model_source=LoadModel(input_model,{"loss":loss})
print("Source model has been loaded")
## create the new model: 
model_new=CreateModels(vocab_size=vocab_size,emb_dim=emb_dim, 
                       num_lstm_units=num_units,batch_size=1, 
                       recurrent_dropout=0,input_dropout=0)
print("New naive model has been created")
## Setting the weights of the old model to the new model:
model_new.set_weights(model_source.get_weights())

print("Weights have been transferred from the trained to the naive model")
model_new.save(output_dir)
print("The Model has been saved")






