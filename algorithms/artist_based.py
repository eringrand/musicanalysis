import numpy as np
import sqlite3 as sqlite
import pandas as pd
import pandas.io.sql as psql
import matplotlib.pyplot as plt
import random

from collections import OrderedDict

import matplotlib
matplotlib.style.use('ggplot')

def apk(actual, predicted, k=500):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 1.0

    return score / min(len(actual), k)

# load observation data
f = open("kaggle_visible_evaluation_triplets.txt", 'rb')
eval = pd.read_csv(f,sep='\t',header = None, names = ['user_id','sid','plays'])

# count number of songs per user and subset
userhist = eval.groupby('user_id').count()
userhist = pd.DataFrame(userhist).reset_index()
usersub = userhist[userhist['plays']>27]

# count number of users per song and subset
songhist = eval.groupby('sid').count()
songhist = pd.DataFrame(songhist).reset_index()
songsub = songhist[songhist['plays']>22]

# subset the whole dataset
sub = eval[eval['sid'].isin(songsub['sid'])]
sub = sub[sub['user_id'].isin(usersub['user_id'])]

# sampling/splitting the dataset
sample = random.sample(sub.index, int(sub.shape[0]*0.2))
trainsub = sub.copy()
trainsub.ix[trainsub.index.isin(sample),'plays'] = 0
testsub = sub.copy()
testsub.ix[~trainsub.index.isin(sample),'plays'] = 0

# generating a matrix out of dataframe
trainpivot = trainsub.pivot(index='user_id',columns='sid', values='plays')
# creating the mapping of user index to user id
user_index = pd.DataFrame(trainpivot.index).reset_index()
user_index.columns = [['user_index','user_id']]
trainsub = pd.merge(trainsub, user_index, on='user_id')
testsub = pd.merge(testsub, user_index, on='user_id')
# creating the mapping of song index to song id
song_index = pd.DataFrame(trainpivot.columns).reset_index()
song_index.columns = [['song_index','sid']]
trainsub = pd.merge(trainsub, song_index, on='sid')
testsub = pd.merge(testsub, song_index, on='sid')

# creating omegau_test
testplays = testsub[testsub.plays>0]
test_usergroup = testsub[testsub.plays>0].groupby('user_index')
omegau_test = {}
for i in list(set(testplays.user_index.values)):
    omegau_test[i] = list(test_usergroup.get_group(i)['song_index'])
# creating omegau_train
trainplays = trainsub[trainsub.plays>0]
train_usergroup = trainplays.groupby('user_index')
omegau_train = {}
for i in list(set(trainplays.user_index.values)):
    omegau_train[i] = list(train_usergroup.get_group(i)['song_index'])


## New code for artist-based recommendation


# Load track metadata to get artist_id
# http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/track_metadata.db
con = sqlite.connect("track_metadata.db")
with con:
    sql = "SELECT * FROM songs"
    track_metadata = psql.read_sql(sql, con)
con.close()

# Renaming columns to match other dataframes
track_metadata=track_metadata.rename(columns = {'track_id':'tid'})
track_metadata=track_metadata.rename(columns = {'song_id':'sid'})
# Subsetting track_metadata to only have songs on the song_index on the subset data
track_metadata=pd.merge(track_metadata,song_index, on='sid')

# Group by songs and sum the amount of plays, then sort in descending amount of plays
eval_songs = eval.groupby('sid').sum().reset_index().sort('plays',ascending=False)[['sid','plays']]
# Subset songs to subset while including track metadata
eval_songmeta = pd.merge(eval_songs, track_metadata, on='sid')

# Generate a list of songs for each artist_id in the order of decreasing popularity of songs
artist_group = eval_songmeta.groupby('artist_id')
artist_songdict = {}
for i in list(set(eval_songmeta.artist_id.values)):
    artist_songdict[i] = list(OrderedDict.fromkeys(artist_group.get_group(i)['song_index']))

# As one song_id can map to multiple track_id, drop all the duplicate song_ids, this is an assumption but the mapping of song_id to artist_id should be unique
track_sid = track_metadata.drop_duplicates('sid')
# Subsetting the user information based on the training set
eval_user = pd.merge(trainsub[trainsub.plays>0], track_sid, on='sid')
# Generate a list of artist_id for each user_index in order of decreasing amount of plays
user_songs = eval_user.groupby(['user_index','artist_id']).sum().reset_index()[['user_index','artist_id','plays']]
user_songs = user_songs.sort(['user_index','plays'], ascending=False)[['user_index','artist_id','plays']]

# Generate a list of song indexes in decreasing order of plays
popularsonglist = list(eval_songmeta.song_index.values)

# apk calculation
apk_sum = 0

for key_user_index in list(set(omegau_test).intersection(omegau_train)):
    songlist = []
    for key_artist_id in list(user_songs[user_songs.user_index==key_user_index].artist_id.values):
        songlist += artist_songdict[key_artist_id]
    # appending the popularsonglist to the generated songlist, assume 600 is more than enough to account for any overlaps
    songlist = list(OrderedDict.fromkeys(songlist + popularsonglist))[:600]
    # starting from np.delete, this function is just to remove the songs in songlist that already exist in omegau_train
    apk_sum += apk(omegau_test[key_user_index],np.delete(songlist,np.nonzero(np.in1d(songlist,omegau_train[key_user_index])))[:500])

map = apk_sum/len(list(set(omegau_test).intersection(omegau_train)))


