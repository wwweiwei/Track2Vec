import numpy as np
import pandas as pd
from reclist.abstractions import RecModel
from gensim.models import Word2Vec
from tqdm import tqdm
import random

class myWord2Vec(RecModel):

    def __init__(self, items, users, top_k: int=100, **kwargs):
        super(myWord2Vec, self).__init__()
        """
        :param top_k: numbers of recommendation to return for each user. Defaults to 20.
        """
        self.items = items
        self.users = self._convert_user_info(users)
        self.top_k = top_k
        self.mappings = None

    def _convert_user_info(self, user_info):
        user_info['gender'] = user_info['gender'].fillna(value='n')
        user_info['country'] = user_info['country'].fillna(value='UNKNOWN')

        return user_info

    def train(self, train_df: pd.DataFrame, **kwargs):
        
        df = train_df[['user_id', 'track_id', 'timestamp']].sort_values('timestamp')
        df = pd.DataFrame(df).join(self.users, on='user_id', how='left')
        p_us = df[df['country'] == 'US'].groupby(['user_id'], sort=False)['track_id'].agg(list)
        p_ru = df[df['country'] == 'RU'].groupby(['user_id'], sort=False)['track_id'].agg(list)
        p_de = df[df['country'] == 'DE'].groupby(['user_id'], sort=False)['track_id'].agg(list)
        p_uk = df[df['country'] == 'UK'].groupby(['user_id'], sort=False)['track_id'].agg(list)
        p_na = df[(df['country'] != 'US') & (df['country'] != 'RU') & (df['country'] != 'DE') & (df['country'] != 'UK')].groupby(['user_id'], sort=False)['track_id'].agg(list)

        self.mymodel_us = Word2Vec(p_us.values.tolist(), min_count=0, vector_size=300, window=40, epochs=10, sg=0, workers=4)
        self.mymodel_ru = Word2Vec(p_ru.values.tolist(), min_count=0, vector_size=300, window=40, epochs=10, sg=0, workers=4)
        self.mymodel_de = Word2Vec(p_de.values.tolist(), min_count=0, vector_size=300, window=40, epochs=10, sg=0, workers=4)
        self.mymodel_uk = Word2Vec(p_uk.values.tolist(), min_count=0, vector_size=300, window=40, epochs=10, sg=0, workers=4)
        self.mymodel_na = Word2Vec(p_na.values.tolist(), min_count=0, vector_size=300, window=40, epochs=10, sg=0, workers=4)

        user_tracks = pd.DataFrame(p_us)
        user_tracks["track_id_sampled"] = user_tracks["track_id"].apply(lambda x : random.choices(x, k=40)) 
        self.mappings = user_tracks.T.to_dict() # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}

        user_tracks = pd.DataFrame(p_ru)
        user_tracks["track_id_sampled"] = user_tracks["track_id"].apply(lambda x : random.choices(x, k=40)) 
        self.mappings.update(user_tracks.T.to_dict()) # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}

        user_tracks = pd.DataFrame(p_de)
        user_tracks["track_id_sampled"] = user_tracks["track_id"].apply(lambda x : random.choices(x, k=40)) 
        self.mappings.update(user_tracks.T.to_dict()) # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}

        user_tracks = pd.DataFrame(p_uk)
        user_tracks["track_id_sampled"] = user_tracks["track_id"].apply(lambda x : random.choices(x, k=40)) 
        self.mappings.update(user_tracks.T.to_dict()) # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}

        user_tracks = pd.DataFrame(p_na)
        user_tracks["track_id_sampled"] = user_tracks["track_id"].apply(lambda x : random.choices(x, k=40)) 
        self.mappings.update(user_tracks.T.to_dict()) # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}

    def predict(self, user_ids: pd.DataFrame) -> pd.DataFrame:
        user_ids = user_ids.copy()
        k = self.top_k
        
        pbar = tqdm(total=len(user_ids), position=0)
        predictions = []
        for user in user_ids["user_id"]:
            user_tracks = self.mappings[user]["track_id_sampled"]
            user_country = self.users.loc[[user]]['country'].values[0]

            if user_country == 'US':
                get_user_embedding = np.mean([self.mymodel_us.wv[t] for t in user_tracks], axis=0)
                max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
                user_predictions = [k[0] for k in self.mymodel_us.wv.most_similar(positive=[get_user_embedding], 
                                                                            topn=max_number_of_returned_items)]
            elif user_country == 'RU':
                get_user_embedding = np.mean([self.mymodel_ru.wv[t] for t in user_tracks], axis=0)
                max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
                user_predictions = [k[0] for k in self.mymodel_ru.wv.most_similar(positive=[get_user_embedding], 
                                                                            topn=max_number_of_returned_items)]
            elif user_country == 'DE':
                get_user_embedding = np.mean([self.mymodel_de.wv[t] for t in user_tracks], axis=0)
                max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
                user_predictions = [k[0] for k in self.mymodel_de.wv.most_similar(positive=[get_user_embedding], 
                                                                            topn=max_number_of_returned_items)]
            elif user_country == 'UK':
                get_user_embedding = np.mean([self.mymodel_uk.wv[t] for t in user_tracks], axis=0)
                max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
                user_predictions = [k[0] for k in self.mymodel_uk.wv.most_similar(positive=[get_user_embedding], 
                                                                            topn=max_number_of_returned_items)]
            else:
                get_user_embedding = np.mean([self.mymodel_na.wv[t] for t in user_tracks], axis=0)
                max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
                user_predictions = [k[0] for k in self.mymodel_na.wv.most_similar(positive=[get_user_embedding], 
                                                                            topn=max_number_of_returned_items)]
            user_predictions = list(filter(lambda x: x not in 
                                           self.mappings[user]["track_id"], user_predictions))[0:self.top_k]
            predictions.append(user_predictions)

            pbar.update(1)
        pbar.close()

        users = user_ids["user_id"].values.reshape(-1, 1)
        predictions = np.concatenate([users, np.array(predictions)], axis=1)
        return pd.DataFrame(predictions, columns=['user_id', *[str(i) for i in range(k)]]).set_index('user_id')