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
# word embeddings load and vectorization


import spacy
import csv
import numpy as np
from tqdm import tqdm


class Vectorizer(object):
    """transform raw text to vector representations"""
    def __init__(self, lang_code):
        print('loading embeddings ...')
        # This only works previous download of spacy data
        # python -m spacy download en_core_web_lg
        # python -m spacy download es
        if lang_code == 'es':            
            self.embeddings = spacy.load('es_core_news_sm')
            print('Usign Spanish embeddings')
        elif lang_code == 'en':
            print('Usign English embeddings')
            self.embeddings = spacy.load('en_core_web_lg')
        else:
            print('unknown lang_code: ', lang_code,' usign English embeddings')
        print('OK')

    def add_embeddings(self, df, target_col, vec_col):
        """insert a column with vectors"""
        print('adding embeddings ...')
        # # integrity of the target column
        df = df.drop_duplicates(subset=target_col)
        df = df.dropna()
        # # append column with embeddings
        vectors = []
        for target_text in tqdm(df[target_col]):
            try:
                vectors.append(self.embeddings(target_text).vector)
            except Exception as e:
                print('Exception vectorizing the target_text:', target_text)
                raise e
        print('type',type(self.embeddings('hola a todos').vector), self.embeddings('hola a todos').vector.dtype)
        df[vec_col] = vectors
        #df[vec_col] = df[vec_col].apply(np.array)
        #df.to_csv('~/repo/eventsmx/df.csv', index=False)
        # df.to_csv('~/repo/eventsmx/df.csv',
        #     quoting=csv.QUOTE_NONNUMERIC,
        #     index=False)
        print('OK')
        return df
