#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:07:58 2019

@author: Hesham El Abd
@description: A Script to generate text from a language model. 
"""
import argparse
import pickle
import numpy as np
from utility_functions import DecodeText,LoadModel,GenerateText

# Processing user inputs
parser=argparse.ArgumentParser(description="""
                        ***Generate a text from a trained language model****
                               """)

parser.add_argument("--model",
                    help="The Path to a saved model in an hf5 format")

parser.add_argument("--output",
                    help="The path save the generated text ")

parser.add_argument("--len", 
                    help="The number of characters to be generated from the"+
                    "model",
                    type=int)

parser.add_argument("--temp", 
                    help="The Sampling temp, a float between zero and 1",
                    type=float
                    )

parser.add_argument("--cond_string",
                    help="A conditionl String to generate text from the model")

user_inputs=parser.parse_args()
path2model=user_inputs.model
output_dir=user_inputs.output
text_len=user_inputs.len
sampling_temp=user_inputs.temp
cond_string=user_inputs.cond_string
## load the models and maps: 
generation_model=LoadModel(path2model)
with open("../Resources/char2idx.pickle","rb") as infile:
    char2idx=pickle.load(infile)
with open("../Resources/idx2char.pickle","rb") as infile:
    idx2char=pickle.load(infile)

## generate the text 
TextGeneratedByModel=GenerateText(model=generation_model,
                  TextLength=text_len, 
                  cond_string=cond_string,
                  sampling_temp=sampling_temp,
                 from_char_to_int_map=char2idx,
                 from_int_to_char_map=idx2char)
## Writing to the outputfile
with open(output_dir,"w") as outfile: 
    outfile.write("With The Condition String:\n"+cond_string+"\n")
    outfile.write("The model generated:\n"+TextGeneratedByModel)

print("The Text has been generated and was written to: \t"+output_dir)


