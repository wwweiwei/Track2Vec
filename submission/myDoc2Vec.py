import numpy as np
import pandas as pd
from reclist.abstractions import RecModel
from gensim.models import Word2Vec
from tqdm import tqdm
import random

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class myDoc2Vec(RecModel):

    def __init__(self, items, users, top_k: int=100, vector_size: int=300, window:int=30, epochs: int=10, **kwargs):
        super(myDoc2Vec, self).__init__()
        """
        :param top_k: numbers of recommendation to return for each user. Defaults to 20.
        """
        self.items = items
        self.users = self._convert_user_info(users)
        self.top_k = top_k
        self.vector_size = vector_size
        self.window = window
        self.epochs = epochs

        self.mappings = None

    def _convert_user_info(self, user_info):
        user_info['gender'] = user_info['gender'].fillna(value='n')
        user_info['country'] = user_info['country'].fillna(value='UNKNOWN')
        user_info['playcount'] = user_info['playcount'].fillna(value=0)

        return user_info


    def train(self, train_df: pd.DataFrame, **kwargs):
        df = train_df[['user_id', 'track_id', 'timestamp', 'user_track_count']].sort_values('timestamp')
        df_trackid = df.groupby(['user_id'], sort=False)['track_id'].agg(list)
        df = pd.DataFrame(df_trackid).join(df.groupby('user_id', as_index=True, sort=False)[['user_track_count']].sum(), on='user_id', how='left')
        self.train_df = df

        p_1 = df[df['user_track_count'] <= 100]['track_id']
        p_2 = df[(100 < df['user_track_count']) & (df['user_track_count'] <= 1000)]['track_id']
        p_3 = df[1000 < df['user_track_count']]['track_id']

        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(p_1.values.tolist())]
        self.mymodel_1 = Doc2Vec(documents, vector_size=self.vector_size, window=self.window, min_count=0, workers=4)
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(p_2.values.tolist())]
        self.mymodel_2 = Doc2Vec(documents, vector_size=self.vector_size, window=self.window, min_count=0, workers=4)
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(p_3.values.tolist())]
        self.mymodel_3 = Doc2Vec(documents, vector_size=self.vector_size, window=self.window, min_count=0, workers=4)

        user_tracks = pd.DataFrame(p_1)
        user_tracks["track_id_sampled"] = user_tracks["track_id"].apply(lambda x : random.choices(x, k=40)) 
        self.mappings = user_tracks.T.to_dict() # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}

        user_tracks = pd.DataFrame(p_2)
        user_tracks["track_id_sampled"] = user_tracks["track_id"].apply(lambda x : random.choices(x, k=40)) 
        self.mappings.update(user_tracks.T.to_dict()) # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}
       
        user_tracks = pd.DataFrame(p_3)
        user_tracks["track_id_sampled"] = user_tracks["track_id"].apply(lambda x : random.choices(x, k=40)) 
        self.mappings.update(user_tracks.T.to_dict()) # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}

    def predict(self, user_ids: pd.DataFrame):
        user_ids = user_ids.copy()
        k = self.top_k
        
        pbar = tqdm(total=len(user_ids), position=0)
        predictions = []
        for user in user_ids["user_id"]:
            user_tracks = self.mappings[user]["track_id_sampled"]
            user_track_count = self.train_df.loc[user]['user_track_count']#.values[0]
            # gender = self.users.loc[[user]]['gender'].values[0]

            if user_track_count <= 100:
                get_user_embedding = np.mean([self.mymodel_1.wv[t] for t in user_tracks], axis=0)
                max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
                user_predictions = [k[0] for k in self.mymodel_1.wv.most_similar(positive=[get_user_embedding], 
                                                                            topn=max_number_of_returned_items)]
            elif 100 < user_track_count and user_track_count <= 1000:
                get_user_embedding = np.mean([self.mymodel_2.wv[t] for t in user_tracks], axis=0)
                max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
                user_predictions = [k[0] for k in self.mymodel_2.wv.most_similar(positive=[get_user_embedding], 
                                                                            topn=max_number_of_returned_items)]
            else:
                get_user_embedding = np.mean([self.mymodel_3.wv[t] for t in user_tracks], axis=0)
                max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
                user_predictions = [k[0] for k in self.mymodel_3.wv.most_similar(positive=[get_user_embedding], 
                                                                            topn=max_number_of_returned_items)]

            user_predictions = list(filter(lambda x: x not in 
                                           self.mappings[user]["track_id"], user_predictions))[0:self.top_k]
            predictions.append(user_predictions)

            pbar.update(1)
        pbar.close()

        users = user_ids["user_id"].values.reshape(-1, 1)
        predictions = np.concatenate([users, np.array(predictions)], axis=1)
        return pd.DataFrame(predictions, columns=['user_id', *[str(i) for i in range(k)]]).set_index('user_id')