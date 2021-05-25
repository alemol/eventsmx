# -*- coding: utf-8 -*-
# 
# This is part of events extractions
#
# Created by Alejandro Molina
# october 2020
# 
# This project is licensed under CC License - see the LICENSE file for details.
# Copyright (c) 2020 Alejandro Molina Villegas
#
# utilities and help functions


import os
from os.path import exists, isdir, isfile, expanduser
import pandas as pd


def load_collection(input_path):
    """determine if dir or file return dataframe anyway"""
    true_path = expanduser(input_path)
    if exists(true_path) and isdir(true_path):
        return load_dir(true_path)
    elif exists(true_path) and isfile(true_path) and true_path.endswith('csv'):
        return load_csv(true_path)
    elif exists(true_path) and isfile(true_path) and true_path.endswith('json'):
        return load_json(true_path)
    return None

def load_json(json_path):
    """read a single json file"""
    print('loading file ...')
    df = pd.read_json(json_path, orient='records', encoding='utf-8')
    print(df.head())
    #df.to_csv('santi.csv')
    print('OK')
    return(df)

def load_csv(csv_path):
    """read a single csv file"""
    print('loading file ...')
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    df = df.dropna()
    print('OK')
    return df

def load_dir(dir_name):
    """read csv files from dir"""
    print('loading dir ...')
    data_frames = []
    dir_path = os.path.expanduser(dir_name)
    for (root, directory, files) in os.walk(dir_path):
        for file in files:
            if file.endswith('csv'):
                full_path = os.path.join(root, file)
                print(file)
                aux_df = pd.read_csv(full_path,
                    encoding='UTF-8',
                    engine='python')
                data_frames.append(aux_df)
    df = pd.concat(data_frames)
    print('OK')
    return df
