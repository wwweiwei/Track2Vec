import numpy as np
import pandas as pd
from reclist.abstractions import RecModel
from gensim.models import Word2Vec
from tqdm import tqdm
import random

class MyModel(RecModel):

    def __init__(self, top_k: int=100, **kwargs):
        super(MyModel, self).__init__()
        """
        :param top_k: numbers of recommendation to return for each user. Defaults to 20.
        """
        self.top_k = top_k
        self.mappings = None

    def train(self, train_df: pd.DataFrame, **kwargs):
        
        # let's put tracks in order so we can build those sentences
        df = train_df[['user_id', 'track_id', 'timestamp']].sort_values('timestamp')
        
        # we group by user and create sequences of tracks. 
        # each row in "track_id" will be a sequence of tracks
        p = df.groupby('user_id', sort=False)['track_id'].agg(list)
        
        # we now build "sentences" : sequences of tracks
        sentences = p.values.tolist()

        # train word2vec:
        # large window and small vector size...lots of epochs
        self.mymodel = Word2Vec(sentences, min_count=2, vector_size=300, window=40, epochs=10, sg=0, workers=3)
        
        user_tracks = pd.DataFrame(p)
        
        # we sample 40 songs for each user. This will be used at runtime to build
        # a user vector
        user_tracks["track_id_sampled"] = user_tracks["track_id"].apply(lambda x : random.choices(x, k=40)) 

        # this dictionary maps users to the songs:
        # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}
        self.mappings = user_tracks.T.to_dict()

    def predict(self, user_ids: pd.DataFrame) -> pd.DataFrame:

        user_ids = user_ids.copy()
        k = self.top_k

       
        pbar = tqdm(total=len(user_ids), position=0)
        
        predictions = []
        
        # probably not the fastest way to do this
        for user in user_ids["user_id"]:
          
          	# for each user we get their sample tracks
            user_tracks = self.mappings[user]["track_id_sampled"]
            
            # average to get user embedding
            get_user_embedding = np.mean([self.mymodel.wv[t] for t in user_tracks], axis=0)
            
            
            # we need to filter out stuff from the user history. We don't want to suggest to the user 
            # something they have already listened to
            max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k

            # let's predict the tracks
            user_predictions = [k[0] for k in self.mymodel.wv.most_similar(positive=[get_user_embedding], 
                                                                           topn=max_number_of_returned_items)]
            # let's filter out songs the user has already listened to
            user_predictions = list(filter(lambda x: x not in 
                                           self.mappings[user]["track_id"], user_predictions))[0:self.top_k]
            
            # append to the return list
            predictions.append(user_predictions)

            pbar.update(1)
        pbar.close()
        
        # lil trick to reconstruct a dataframe that has user ids as index and the predictions as columns
        # This is a very important part! consistency in how you return the results is fundamental for the 
        # evaluation
     
        users = user_ids["user_id"].values.reshape(-1, 1)
        predictions = np.concatenate([users, np.array(predictions)], axis=1)
        return pd.DataFrame(predictions, columns=['user_id', *[str(i) for i in range(k)]]).set_index('user_id')