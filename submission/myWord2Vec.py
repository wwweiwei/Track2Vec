import numpy as np
import pandas as pd
from reclist.abstractions import RecModel
from gensim.models import Word2Vec
from tqdm import tqdm
import random

class myWord2Vec(RecModel):

    def __init__(self, items, users, top_k: int=100, vector_size: int=300, window:int=30, epochs: int=10, **kwargs):
        super(myWord2Vec, self).__init__()
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
        
        df = train_df[['user_id', 'track_id', 'timestamp']].sort_values('timestamp')
        df = pd.DataFrame(df).join(self.users, on='user_id', how='left')
        p_1 = df[df['playcount'] <= 10].groupby(['user_id'], sort=False)['track_id'].agg(list)
        p_2 = df[(10 < df['playcount']) & (df['playcount'] <= 100)].groupby(['user_id'], sort=False)['track_id'].agg(list)
        p_3_m = df[(100 < df['playcount']) & (df['playcount'] <= 1000) & (df['gender'] == 'm')].groupby(['user_id'], sort=False)['track_id'].agg(list)
        p_3_f = df[(100 < df['playcount']) & (df['playcount'] <= 1000) & (df['gender'] == 'f')].groupby(['user_id'], sort=False)['track_id'].agg(list)
        p_3_n = df[(100 < df['playcount']) & (df['playcount'] <= 1000) & (df['gender'] != 'm') & (df['gender'] != 'f')].groupby(['user_id'], sort=False)['track_id'].agg(list)
        p_4 = df[1000 < df['playcount']].groupby(['user_id'], sort=False)['track_id'].agg(list)
        # p_4_m = df[(1000 < df['playcount']) & (df['gender'] == 'm')].groupby(['user_id'], sort=False)['track_id'].agg(list)
        # p_4_f = df[(1000 < df['playcount']) & (df['gender'] == 'f')].groupby(['user_id'], sort=False)['track_id'].agg(list)
        # p_4_n = df[(1000 < df['playcount']) & (df['gender'] != 'm') & (df['gender'] != 'f')].groupby(['user_id'], sort=False)['track_id'].agg(list)

        self.mymodel_1 = Word2Vec(p_1.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, epochs=self.epochs, sg=0, workers=4)
        self.mymodel_2 = Word2Vec(p_2.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, epochs=self.epochs, sg=0, workers=4)
        self.mymodel_3_m = Word2Vec(p_3_m.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, epochs=self.epochs, sg=0, workers=4)
        self.mymodel_3_f = Word2Vec(p_3_f.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, epochs=self.epochs, sg=0, workers=4)
        self.mymodel_3_n = Word2Vec(p_3_n.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, epochs=self.epochs, sg=0, workers=4)
        self.mymodel_4 = Word2Vec(p_4.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, epochs=self.epochs, sg=0, workers=4)
        # self.mymodel_4_m = Word2Vec(p_4_m.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, epochs=self.epochs, sg=0, workers=4)
        # self.mymodel_4_f = Word2Vec(p_4_f.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, epochs=self.epochs, sg=0, workers=4)
        # self.mymodel_4_n = Word2Vec(p_4_n.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, epochs=self.epochs, sg=0, workers=4)

        user_tracks = pd.DataFrame(p_1)
        user_tracks["track_id_sampled"] = user_tracks["track_id"].apply(lambda x : random.choices(x, k=40)) 
        self.mappings = user_tracks.T.to_dict() # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}

        user_tracks = pd.DataFrame(p_2)
        user_tracks["track_id_sampled"] = user_tracks["track_id"].apply(lambda x : random.choices(x, k=40)) 
        self.mappings.update(user_tracks.T.to_dict()) # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}

        user_tracks = pd.DataFrame(p_3_m)
        user_tracks["track_id_sampled"] = user_tracks["track_id"].apply(lambda x : random.choices(x, k=40)) 
        self.mappings.update(user_tracks.T.to_dict()) # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}

        user_tracks = pd.DataFrame(p_3_f)
        user_tracks["track_id_sampled"] = user_tracks["track_id"].apply(lambda x : random.choices(x, k=40)) 
        self.mappings.update(user_tracks.T.to_dict()) # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}

        user_tracks = pd.DataFrame(p_3_n)
        user_tracks["track_id_sampled"] = user_tracks["track_id"].apply(lambda x : random.choices(x, k=40)) 
        self.mappings.update(user_tracks.T.to_dict()) # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}

        user_tracks = pd.DataFrame(p_4)
        user_tracks["track_id_sampled"] = user_tracks["track_id"].apply(lambda x : random.choices(x, k=40)) 
        self.mappings.update(user_tracks.T.to_dict()) # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}

        # user_tracks = pd.DataFrame(p_4_m)
        # user_tracks["track_id_sampled"] = user_tracks["track_id"].apply(lambda x : random.choices(x, k=40)) 
        # self.mappings.update(user_tracks.T.to_dict()) # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}

        # user_tracks = pd.DataFrame(p_4_f)
        # user_tracks["track_id_sampled"] = user_tracks["track_id"].apply(lambda x : random.choices(x, k=40)) 
        # self.mappings.update(user_tracks.T.to_dict()) # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}

        # user_tracks = pd.DataFrame(p_4_n)
        # user_tracks["track_id_sampled"] = user_tracks["track_id"].apply(lambda x : random.choices(x, k=40)) 
        # self.mappings.update(user_tracks.T.to_dict()) # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}

    def predict(self, user_ids: pd.DataFrame) -> pd.DataFrame:
        user_ids = user_ids.copy()
        k = self.top_k
        
        pbar = tqdm(total=len(user_ids), position=0)
        predictions = []
        for user in user_ids["user_id"]:
            user_tracks = self.mappings[user]["track_id_sampled"]
            user_playcount = self.users.loc[[user]]['playcount'].values[0]
            gender = self.users.loc[[user]]['gender'].values[0]

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
            elif 100 < user_playcount and user_playcount <= 1000 and gender == 'm':
                get_user_embedding = np.mean([self.mymodel_3_m.wv[t] for t in user_tracks], axis=0)
                max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
                user_predictions = [k[0] for k in self.mymodel_3_m.wv.most_similar(positive=[get_user_embedding], 
                                                                            topn=max_number_of_returned_items)]
            elif 100 < user_playcount and user_playcount <= 1000 and gender == 'f':
                get_user_embedding = np.mean([self.mymodel_3_f.wv[t] for t in user_tracks], axis=0)
                max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
                user_predictions = [k[0] for k in self.mymodel_3_f.wv.most_similar(positive=[get_user_embedding], 
                                                                            topn=max_number_of_returned_items)]
            elif 100 < user_playcount and user_playcount <= 1000 and gender != 'm' and gender != 'f':
                get_user_embedding = np.mean([self.mymodel_3_n.wv[t] for t in user_tracks], axis=0)
                max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
                user_predictions = [k[0] for k in self.mymodel_3_n.wv.most_similar(positive=[get_user_embedding], 
                                                                            topn=max_number_of_returned_items)]
            else:
                get_user_embedding = np.mean([self.mymodel_4.wv[t] for t in user_tracks], axis=0)
                max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
                user_predictions = [k[0] for k in self.mymodel_4.wv.most_similar(positive=[get_user_embedding], 
                                                                            topn=max_number_of_returned_items)]
            # elif 1000 < user_playcount and gender == 'm':
            #     get_user_embedding = np.mean([self.mymodel_4_m.wv[t] for t in user_tracks], axis=0)
            #     max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
            #     user_predictions = [k[0] for k in self.mymodel_4_m.wv.most_similar(positive=[get_user_embedding], 
            #                                                                 topn=max_number_of_returned_items)]
            # elif 1000 < user_playcount and gender == 'f':
            #     get_user_embedding = np.mean([self.mymodel_4_f.wv[t] for t in user_tracks], axis=0)
            #     max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
            #     user_predictions = [k[0] for k in self.mymodel_4_f.wv.most_similar(positive=[get_user_embedding], 
            #                                                                 topn=max_number_of_returned_items)]
            # else:
            #     get_user_embedding = np.mean([self.mymodel_4_n.wv[t] for t in user_tracks], axis=0)
            #     max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
            #     user_predictions = [k[0] for k in self.mymodel_4_n.wv.most_similar(positive=[get_user_embedding], 
            #                                                                 topn=max_number_of_returned_items)]

            user_predictions = list(filter(lambda x: x not in 
                                           self.mappings[user]["track_id"], user_predictions))[0:self.top_k]
            predictions.append(user_predictions)

            pbar.update(1)
        pbar.close()

        users = user_ids["user_id"].values.reshape(-1, 1)
        predictions = np.concatenate([users, np.array(predictions)], axis=1)
        return pd.DataFrame(predictions, columns=['user_id', *[str(i) for i in range(k)]]).set_index('user_id')