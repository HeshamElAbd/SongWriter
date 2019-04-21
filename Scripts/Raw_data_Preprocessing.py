#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 03:39:13 2019
@author: Hesham El Abd
@Description: The script Reads the songlyrics dataset, extract the songs, 
processes them then write it to a file.
The data was downladed from Sergey Kuznetsov @ Kaggle
check it out @: 
    https://www.kaggle.com/mousehead/songlyrics/version/1#
"""
# import modules:
import pandas as pd 
import os

# read the data:
raw_table=pd.read_csv("../Data/songlyrics/songdata.csv")
raw_text=raw_table.text.to_list()

# define the start and end tokens for each song.
start_token="$ " 
end_token=" % " 
# add the toekns to add the words in the model: 
song_list=[]
for raw_song in raw_text:
    song_list.append(start_token+raw_song+end_token)

# generating the text corpa: 
song_corpa=""
for song in song_list:
    for char in song:
        song_corpa+=char

# writing it down: 
with open("../Data/song_corpa.txt","w") as outfile: 
        outfile.write(song_corpa)






