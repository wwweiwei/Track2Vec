import numpy as np
import pandas as pd
from reclist.abstractions import RecModel
from gensim.models import Word2Vec
from tqdm import tqdm
import random

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import Counter
import itertools
import time

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

seed = 27
set_random_seed(seed)

class myWord2Vec(RecModel):

    def __init__(self, items, users, top_k: int=100, vector_size: int=300, window:int=30, epochs: int=10, negative: int=5, **kwargs):
        super(myWord2Vec, self).__init__()
        """
        :param top_k: numbers of recommendation to return for each user. Defaults to 20.
        """
        self.items = items
        self.users = self._convert_user_info(users)
        self.top_k = top_k
        self.mappings = None
        self.window = window
        self.epochs = epochs
        self.negative = negative
        self.vector_size = vector_size

    def _convert_user_info(self, user_info):
        user_info['gender'] = user_info['gender'].fillna(value='n')
        # user_info['country'] = user_info['country'].fillna(value='UNKNOWN')
        user_info['playcount'] = user_info['playcount'].fillna(value=0)

        return user_info

    def train_playcount(self, df):
        p_1 = df[df['playcount'] <= 10].groupby(['user_id'], sort=False)['track_id'].agg(list)
        p_2 = df[(10 < df['playcount']) & (df['playcount'] <= 100)].groupby(['user_id'], sort=False)['track_id'].agg(list)
        p_3 = df[(100 < df['playcount']) & (df['playcount'] <= 1000)].groupby(['user_id'], sort=False)['track_id'].agg(list)
        p_4 = df[1000 < df['playcount']].groupby(['user_id'], sort=False)['track_id'].agg(list)
        
        # def hash(astring):
        #     return ord(astring[0])

        self.mymodel_1 = Word2Vec(p_1.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, negative=self.negative, epochs=self.epochs, sg=0, workers=4) #, hashfxn=hash)
        self.mymodel_2 = Word2Vec(p_2.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, negative=self.negative, epochs=self.epochs, sg=0, workers=4) #, hashfxn=hash)
        self.mymodel_3 = Word2Vec(p_3.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, negative=self.negative, epochs=self.epochs, sg=0, workers=4) #, hashfxn=hash)
        self.mymodel_4 = Word2Vec(p_4.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, negative=self.negative, epochs=self.epochs, sg=0, workers=4) #, hashfxn=hash)

        p = p_1.append(p_2).append(p_3).append(p_4)

        user_tracks = pd.DataFrame(p)
        user_tracks['playcount_track_id_sampled'] = user_tracks['track_id'].apply(lambda x : random.choices(x, k=40)) 
        self.mappings = user_tracks.T.to_dict() # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}


    def train_gender(self, df):
        p_m = df[df['gender'] == 'm'].groupby(['user_id'], sort=False)['track_id'].agg(list)
        p_f = df[df['gender'] == 'f'].groupby(['user_id'], sort=False)['track_id'].agg(list)
        p_n = df[(df['gender'] != 'm') & (df['gender'] != 'f')].groupby(['user_id'], sort=False)['track_id'].agg(list)

        self.mymodel_m = Word2Vec(p_m.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, negative=self.negative, epochs=self.epochs, sg=0, workers=4)
        self.mymodel_f = Word2Vec(p_f.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, negative=self.negative, epochs=self.epochs, sg=0, workers=4)
        self.mymodel_n = Word2Vec(p_n.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, negative=self.negative, epochs=self.epochs, sg=0, workers=4)
        
        p = p_m.append(p_f).append(p_n)

        user_tracks = pd.DataFrame(p)
        user_tracks['gender_track_id_sampled'] = user_tracks['track_id'].apply(lambda x : random.choices(x, k=40))
        gender_dict = user_tracks.T.to_dict()
        for key in self.mappings.keys():
            self.mappings[key]['gender_track_id_sampled'] = gender_dict[key]['gender_track_id_sampled']

    def train_user_track_count(self, df):
        df_trackid = df.groupby(['user_id'], sort=False)['track_id'].agg(list)
        df = pd.DataFrame(df_trackid).join(df.groupby('user_id', as_index=True, sort=False)[['user_track_count']].sum(), on='user_id', how='left')
        self.train_df = df
        df = pd.DataFrame(df).join(self.users, on='user_id', how='left')

        p_1 = df[df['user_track_count'] <= 100]['track_id']
        p_2 = df[(100 < df['user_track_count']) & (df['user_track_count'] <= 1000)]['track_id']
        p_3 = df[1000 < df['user_track_count']]['track_id']

        self.mymodel_utc1 = Word2Vec(p_1.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, negative=self.negative, epochs=self.epochs, sg=0, workers=4)
        self.mymodel_utc2 = Word2Vec(p_2.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, negative=self.negative, epochs=self.epochs, sg=0, workers=4)
        self.mymodel_utc3 = Word2Vec(p_3.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, negative=self.negative, epochs=self.epochs, sg=0, workers=4)

        # documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(p_1.values.tolist())]
        # self.mymodel_utc1 = Doc2Vec(documents, vector_size=self.vector_size, window=self.window, min_count=0, workers=4)
        # documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(p_2.values.tolist())]
        # self.mymodel_utc2 = Doc2Vec(documents, vector_size=self.vector_size, window=self.window, min_count=0, workers=4)
        # documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(p_3.values.tolist())]
        # self.mymodel_utc3 = Doc2Vec(documents, vector_size=self.vector_size, window=self.window, min_count=0, workers=4)

        p = p_1.append(p_2).append(p_3)

        user_tracks = pd.DataFrame(p)
        user_tracks['utc_track_id_sampled'] = user_tracks['track_id'].apply(lambda x : random.choices(x, k=40))
        utc_dict = user_tracks.T.to_dict()
        for key in self.mappings.keys():
            self.mappings[key]['utc_track_id_sampled'] = utc_dict[key]['utc_track_id_sampled']

    def train(self, train_df: pd.DataFrame, **kwargs):
        df = train_df[['user_id', 'track_id', 'timestamp', 'user_track_count']].sort_values('timestamp')
        df = pd.DataFrame(df).join(self.users, on='user_id', how='left')
        
        ts1 = time.time()
        self.train_playcount(df)
        ts2 = time.time()
        print('playcount: ', str(ts2-ts1))
        self.train_gender(df)
        ts3 = time.time()
        print('gender: ', str(ts3-ts2))
        self.train_user_track_count(df)
        ts4 = time.time()
        print('utc: ', str(ts4-ts3))


    def pred_playcount(self, user, user_playcount, user_tracks):
        if user_playcount <= 10:
            get_user_embedding = np.mean([self.mymodel_1.wv[t] for t in user_tracks], axis=0)
            max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
            user_predictions = [k[0] for k in self.mymodel_1.wv.most_similar(positive=[get_user_embedding], 
                                                                        topn=max_number_of_returned_items)]
        elif 10 < user_playcount and user_playcount <= 100:
            get_user_embedding = np.mean([self.mymodel_2.wv[t] for t in user_tracks], axis=0)
            max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
            user_predictions = [k[0] for k in self.mymodel_2.wv.most_similar(positive=[get_user_embedding], 
                                                                        topn=max_number_of_returned_items)]
        elif 100 < user_playcount and user_playcount <= 1000:
            get_user_embedding = np.mean([self.mymodel_3.wv[t] for t in user_tracks], axis=0)
            max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
            user_predictions = [k[0] for k in self.mymodel_3.wv.most_similar(positive=[get_user_embedding], 
                                                                        topn=max_number_of_returned_items)]
        else:
            get_user_embedding = np.mean([self.mymodel_4.wv[t] for t in user_tracks], axis=0)
            max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
            user_predictions = [k[0] for k in self.mymodel_4.wv.most_similar(positive=[get_user_embedding], 
                                                                        topn=max_number_of_returned_items)]

        user_predictions = list(filter(lambda x: x not in 
                                        self.mappings[user]["track_id"], user_predictions))[0:self.top_k]
        return user_predictions

    def pred_gender(self, user, user_gender, user_tracks):
        if user_gender == 'm':
            get_user_embedding = np.mean([self.mymodel_m.wv[t] for t in user_tracks], axis=0)
            max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k 
            user_predictions = [k[0] for k in self.mymodel_m.wv.most_similar(positive=[get_user_embedding], 
                                                                        topn=max_number_of_returned_items)]
        elif user_gender == 'f':
            get_user_embedding = np.mean([self.mymodel_f.wv[t] for t in user_tracks], axis=0)
            max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k
            user_predictions = [k[0] for k in self.mymodel_f.wv.most_similar(positive=[get_user_embedding], 
                                                                        topn=max_number_of_returned_items)]
        else:
            get_user_embedding = np.mean([self.mymodel_n.wv[t] for t in user_tracks], axis=0)
            max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k
            user_predictions = [k[0] for k in self.mymodel_n.wv.most_similar(positive=[get_user_embedding], 
                                                                        topn=max_number_of_returned_items)]

        user_predictions = list(filter(lambda x: x not in 
                                        self.mappings[user]["track_id"], user_predictions))[0:self.top_k]
        return user_predictions

    def pred_user_track_count(self, user, user_track_count, user_tracks):
        if user_track_count <= 100:
            get_user_embedding = np.mean([self.mymodel_utc1.wv[t] for t in user_tracks], axis=0)
            max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
            user_predictions = [k[0] for k in self.mymodel_utc1.wv.most_similar(positive=[get_user_embedding], 
                                                                        topn=max_number_of_returned_items)]
        elif 100 < user_track_count and user_track_count <= 1000:
            get_user_embedding = np.mean([self.mymodel_utc2.wv[t] for t in user_tracks], axis=0)
            max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
            user_predictions = [k[0] for k in self.mymodel_utc2.wv.most_similar(positive=[get_user_embedding], 
                                                                        topn=max_number_of_returned_items)]
        else:
            get_user_embedding = np.mean([self.mymodel_utc3.wv[t] for t in user_tracks], axis=0)
            max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
            user_predictions = [k[0] for k in self.mymodel_utc3.wv.most_similar(positive=[get_user_embedding], 
                                                                        topn=max_number_of_returned_items)]

        user_predictions = list(filter(lambda x: x not in 
                                        self.mappings[user]["track_id"], user_predictions))[0:self.top_k]
        return user_predictions

    def ensemble(self, pred_1, pred_2, pred_3):
        all_pred = list(itertools.chain(pred_1, pred_2, pred_3))
        counter = Counter(all_pred)
        print('counter length: ', len(counter))
        counter_top_k = counter.most_common(self.top_k)
        pred = []
        for tuple in counter_top_k:
            pred.append(tuple[0])

        return pred

    def predict(self, user_ids: pd.DataFrame):
        ts5 = time.time()
        user_ids = user_ids.copy()
        k = self.top_k
        
        pbar = tqdm(total=len(user_ids), position=0)
        predictions = []
        for user in user_ids['user_id']:
            user_tracks_playcount = self.mappings[user]['playcount_track_id_sampled']
            user_tracks_gender = self.mappings[user]['gender_track_id_sampled']
            user_tracks_utc = self.mappings[user]['utc_track_id_sampled']

            user_playcount = self.users.loc[[user]]['playcount'].values[0]
            user_gender = self.users.loc[[user]]['gender'].values[0]
            user_track_count = self.train_df.loc[user]['user_track_count']

            pred_1 = self.pred_playcount(user, user_playcount, user_tracks_playcount)
            pred_2 = self.pred_gender(user, user_gender, user_tracks_gender)
            pred_3 = self.pred_user_track_count(user, user_track_count, user_tracks_utc)

            user_predictions = self.ensemble(pred_1, pred_2, pred_3)
            predictions.append(user_predictions)

            pbar.update(1)
        pbar.close()

        users = user_ids["user_id"].values.reshape(-1, 1)
        predictions = np.concatenate([users, np.array(predictions)], axis=1)

        ts6 = time.time()
        print('pred: ', str(ts6-ts5))
        return pd.DataFrame(predictions, columns=['user_id', *[str(i) for i in range(k)]]).set_index('user_id')
