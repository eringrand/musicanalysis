import numpy as np
import pandas as pd
import random

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

# generating the list of songs ordered by number of counts and number of plays
groupcount = trainsub[trainsub.plays>0].groupby('song_index').count().sort('plays', ascending=False).reset_index()
topcounts = groupcount.song_index.values
groupsum = trainsub[trainsub.plays>0].groupby('song_index').sum().sort('plays', ascending=False).reset_index()
topplays = groupsum.song_index.values

# calculating map for songs by number of counts
apk_sum = 0
for i in omegau_test:
    if i in omegau_train:
        apk_sum += apk(omegau_test[i],np.delete(topcounts,np.nonzero(np.in1d(topcounts,omegau_train[i])))[:500])
    else:
        apk_sum += apk(omegau_test[i],topcounts[:500])

apk_sum/len(omegau_test)

# calculating map for songs by number of plays
apk_sum = 0
for i in omegau_test:
    if i in omegau_train:
        apk_sum += apk(omegau_test[i],np.delete(topplays,np.nonzero(np.in1d(topplays,omegau_train[i])))[:500])
    else:
        apk_sum += apk(omegau_test[i],topplays[:500])

apk_sum/len(omegau_test)