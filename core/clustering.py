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
# clustering data by vector distances


import numpy as np
from sklearn.cluster import DBSCAN


class EmbeddingsClusterer(DBSCAN):
    """cluster similar embeddings"""
    def __init__(self, df):
        self.df = df

    def fit_clusters(self, target_col, epsilon):
        """unsupervised learning to form clusters from data"""
        print('fitting clusters ...')
        # The epsilon parameter determines the maximum
        # distance between two samples for them to be 
        # considered as in the same neighborhood, 
        # meaning that if eps is too big,
        # fewer clusters will be formed, but also if
        # it’s too small, most of the points will be
        # classified as not belonging to a cluster (-1),
        # which will result in a few clusters as well.
        #
        X = np.array(self.df[target_col].tolist())
        print(type(X), X.shape)
        dbscan = DBSCAN(eps=epsilon, min_samples=2, metric='cosine').fit(X)
        print(type(dbscan.labels_), dbscan.labels_.shape, dbscan.labels_)
        print('OK')
        return dbscan.labels_

    def get_clusterk(self, k, date_col, labels_col):
        print('getting cluster ...')
        if not 'label' in self.df.columns:
            print('"label" column is missing')
        cluster_k = self.df[self.df[labels_col] == k]
        cluster_k = cluster_k.sort_values(by=date_col).dropna()
        print('OK')
        return cluster_k
