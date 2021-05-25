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
# events extraction from textual data


from vectorization import Vectorizer
from clustering import EmbeddingsClusterer
from utils import load_collection
from sklearn import metrics
from sklearn.metrics.pairwise import paired_distances
import pandas as pd
import numpy as np
import simplejson as json
import matplotlib.pyplot as plt


class EventsExtractor(object):
    """Extracts events from text"""
    def __init__(self, data, date_col, title_col, text_col, lang='en', vec_col=None,):
        super(EventsExtractor, self).__init__()
        self.date_col = date_col
        self.title_col = title_col
        self.text_col = text_col
        self.target_col = text_col
        self.df = data
        self.df['event_date'] = self.df[self.date_col]
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        if not vec_col:
            self.vec_col = 'vector'
            self.vectorizer = Vectorizer(lang)
            self.df = self.vectorizer.add_embeddings(self.df, self.target_col, self.vec_col)
            self.df[self.vec_col].to_numpy()
        else:
            self.vec_col = vec_col
            self.df[self.vec_col] = self.load_vecs(data, vec_col)
        self.clusterer = EmbeddingsClusterer(self.df)

    def load_vecs(self, df, vec_col):
        print(df.dtypes)
        #df[vec_col]= df[vec_col].str.replace('\n', ' ', inplace=True)
        #df[vec_col]= df[vec_col].str.replace(to_replace=r' +', value=', ', inplace=True)
        #df = df.astype({vec_col: str})
        #df = df.astype({vec_col: np.ndarray})
        df[vec_col] = df[vec_col].to_numpy()
        #print(df.dtypes)
        print(df.head())
        return

    def explore(self, low_val, high_val, step ,label_col='label'):
        n_clases = {}
        silhouette = {}
        for i in np.arange(low_val, high_val, step):            
            self.df[label_col] = self.clusterer.fit_clusters(self.vec_col, i)
            self.n_clusters = self.df[label_col].nunique() - 1
            n_clases.update({i: self.n_clusters})
            ###
            embeddings = self.df[self.vec_col]
            arr_embeddings = embeddings.to_numpy()
            arr_embeddings = np.stack(arr_embeddings)
            labels = self.df[label_col].to_numpy()
            try:
                score = metrics.silhouette_score(arr_embeddings, labels, metric='cosine')
            except Exception as e:
                score = -1.0
            ###
            print('epsilon = {}, n_clusters = {}, silhouette = {}'.format(i, self.n_clusters, score))
            silhouette.update({i: score})
        # Make a plot
        print('vetors shape', arr_embeddings.shape)
        print('labels shape', labels.shape)
        sil_series = pd.Series(silhouette)
        k_series = pd.Series(n_clases)
        # plot
        plt.figure(figsize=(9.9, 6))
        plt.subplot(211)
        k_series.plot()
        plt.title('Classes')
        plt.subplot(212)
        sil_series.plot()
        plt.title('Silhouette')
        plt.savefig('out/clustering_explore.png')

    def extract(self, epsilon, date_col, label_col='label'):
        self.df[label_col] = self.clusterer.fit_clusters(self.vec_col, epsilon)
        # BDSCAN assigns -1 to noise data so there is one special label
        self.n_clusters = self.df[label_col].nunique() - 1
        print('self.n_clusters', self.n_clusters)
        for k in np.arange(0, self.n_clusters, 1):
            if k > -1:
                print('cluster k:', k)
                k_cluster = self.df[self.df[label_col] == k]
                # k_cluster = k_cluster.sort_values(by=date_col).dropna()
                print('shape', k_cluster.shape)
                print(k_cluster.head())
                #k_cluster[['date','labels','text']].to_csv('~/repo/eventsmx/out/clust{}.csv'.format(k))
                #self.clusters.append((k, centroid, k_cluster))
                #centroid = self.get_mean_vector(k_cluster)
                #print('centroid:', centroid)
                #(k, centroid, k_events) = self.get_events(k_cluster)
                e = self.get_events(k_cluster)
                print('events k:', k)
                print(e.shape)
                print(e)
                yield e

    def get_events(self, df, period='D'):
        """organize events extracted from news using only one event by frecuency period, e.g. day, month"""
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        df = df.set_index(self.date_col)
        events = []
        for group_name, df_group in df.groupby(pd.Grouper(freq=period)):
            if df_group.empty:
                continue
            if df_group.shape[0] > 1:
                print('colapsing ',df_group.shape[0], 'events found for ',group_name,' using period criteria:',period)
                colapsed_df =self.colapse_events(df_group)
                events.append(colapsed_df)
            else:
                events.append(df_group)
        df_events = pd.concat(events)
        df_events.sort_values(by=[self.date_col], inplace=True, ascending=True)
        return df_events

    def colapse_events(self, df):
        """colapse cluster getting only one event by period"""
        X = np.array(df[self.vec_col].tolist())
        (n, m) = X.shape
        centroid = self.get_mean_vector(df)
        print(X)
        Y = np.repeat(np.array([centroid]), n, axis=0)
        print(Y)
        D = paired_distances(X, Y, metric='cosine')
        print('Distances')
        print(D)
        index_min = (np.where(D == np.amin(D)))[0]
        print('index_min', index_min)
        return df.iloc[index_min,:]

    def get_mean_vector(self, df):
        """Asses the clusters mean vector, i.e. centroid"""
        X = np.array(df[self.vec_col].tolist())
        m = np.mean(X, axis=0)
        return m


if __name__ == '__main__':
    #DATA_PATH = '~/repo/eventsmx/data/yuc_la_jornada_maya.csv'
    #DATA_PATH = '~/repo/eventsmx/data/septiembre/'
    DATA_PATH = '~/repo/eventsmx/data/septiembre_min'
    data = load_collection(DATA_PATH)
    extractor = EventsExtractor(data, 'date', 'header', 'text', lang='es', vec_col=None)
    extractor.explore(0.001, 0.025, 0.001)
    # for i, e in enumerate(extractor.extract(0.01, 'date', label_col='label')):
    #     e_as_json = json.loads(e.to_json(orient="records"))
    #     with open('out/e_{}.csv'.format(i), 'w') as f:
    #         f.write(json.dumps(e_as_json, ensure_ascii=False, encoding='UTF-8', indent=2))

