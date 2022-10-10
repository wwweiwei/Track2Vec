import os
from datetime import datetime
from dotenv import load_dotenv

# import env variables from file
load_dotenv('upload.env', verbose=True)

# variables for the submission
EMAIL = os.getenv('EMAIL')  # the e-mail you used to sign up
assert EMAIL != '' and EMAIL is not None
BUCKET_NAME = os.getenv('BUCKET_NAME')  # you received it in your e-mail
PARTICIPANT_ID = os.getenv('PARTICIPANT_ID')  # you received it in your e-mail
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')  # you received it in your e-mail
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')  # you received it in your e-mail

# set hyperparameters for Track2Vec
vector_size = 100
epoch = 10
top_k = 100
window = 60
seed = 27
negative = 5
print(f'vector_size: {vector_size} | epoch: {epoch} | top_k: {top_k} | window: {window} | seed: {seed} | ns: {negative}')

# run the evaluation loop when the script is called directly
if __name__ == '__main__':
    # import the basic classes
    from evaluation.EvalRSRunner import EvalRSRunner
    from evaluation.EvalRSRunner import ChallengeDataset
    from submission.Track2Vec import Track2Vec
    print('\n==== Starting evaluation script at: {} ====\n'.format(datetime.utcnow()))
    # load the dataset
    print('\n==== Loading dataset at: {} ====\n'.format(datetime.utcnow()))
    # this will load the dataset with the default values for the challenge
    dataset = ChallengeDataset(seed = seed, num_folds = 4)
    print('\n==== Init runner at: {} ====\n'.format(datetime.utcnow()))
    # run the evaluation loop
    runner = EvalRSRunner(
        dataset = dataset,
        aws_access_key_id = AWS_ACCESS_KEY,
        aws_secret_access_key = AWS_SECRET_KEY,
        participant_id = PARTICIPANT_ID,
        bucket_name = BUCKET_NAME,
        email = EMAIL
        )
    print('==== Runner loaded, starting loop at: {} ====\n'.format(datetime.utcnow()))
    my_model = Track2Vec(
        items = dataset.df_tracks,
        users = dataset.df_users,
        top_k = top_k,
        vector_size = vector_size,
        window = window,
        epochs = epoch,
        negative = negative
    )
    runner.evaluate(
        model = my_model,
        upload = False
        )
    print('\n\n== Evaluation ended at: {} ===='.format(datetime.utcnow()))