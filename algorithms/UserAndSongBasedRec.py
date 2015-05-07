from operator import itemgetter
import numpy as np
from collections import defaultdict
import pandas as pd
import random






f = open("kaggle_visible_evaluation_triplets.txt", 'rb')
eval = pd.read_csv(f,sep='\t',header = None, names = ['user_id','sid','plays'])

userhist = eval.groupby('user_id').count()
userhist = pd.DataFrame(userhist).reset_index()
usersub = userhist[userhist['plays']>27]

songhist = eval.groupby('sid').count()
songhist = pd.DataFrame(songhist).reset_index()
songsub = songhist[songhist['plays']>22]

sub = eval[eval['sid'].isin(songsub['sid'])]
sub = sub[sub['user_id'].isin(usersub['user_id'])]

len(set(sub['user_id']))
len(set(sub['sid']))

sub.shape[0]/float(eval.shape[0])

sub.shape[0]/float(len(set(sub['user_id']))*len(set(sub['sid'])))

#sub_max = pd.DataFrame(sub.groupby('sid').max()).reset_index()
#merged = pd.merge(sub,sub_max,on="sid")
#merged['plays_x'] = merged['plays_x']/merged['plays_y']
#sub_norm = merged[['user_id_x', 'sid', 'plays_x']]
#sub_norm.columns = ['user_id', 'sid', 'plays']

sample = random.sample(sub.index, int(sub.shape[0]*0.2))

trainsub = sub.copy()
trainsub.ix[trainsub.index.isin(sample),'plays'] = 0

testsub = sub.copy()
testsub.ix[~trainsub.index.isin(sample),'plays'] = 0

trainpivot = trainsub.pivot(index='user_id',columns='sid', values='plays')

user_index = pd.DataFrame(trainpivot.index).reset_index()
user_index.columns = [['user_index','user_id']]
trainsub = pd.merge(trainsub, user_index, on='user_id')
testsub = pd.merge(testsub, user_index, on='user_id')
song_index = pd.DataFrame(trainpivot.columns).reset_index()
song_index.columns = [['song_index','sid']]
trainsub = pd.merge(trainsub, song_index, on='sid')
testsub = pd.merge(testsub, song_index, on='sid')

M_train = trainpivot.as_matrix()
M_train = np.nan_to_num(M_train)

rowsum = np.sum(M_train,axis=1).reshape((-1,1))
rowsum[rowsum == 0] = 1E-16
M_train_norm = M_train/rowsum


idftrain = trainsub.copy()
idftrain.plays = 1
idf_train = idftrain.pivot(index='user_id',columns='sid', values='plays').as_matrix()
idf_train = np.nan_to_num(idf_train)
colsum = np.sum(idf_train,axis=0)
idf = np.log10(idf_train.shape[0]/colsum)
M_train_tfidf = M_train_norm*idf

omega = [tuple(x) for x in trainsub[['user_index','song_index']].values]
omega_test = [tuple(x) for x in testsub[['user_index','song_index']].values]

testplay = testsub[testsub.plays>0]
test_usergroup = testplay.groupby('user_index')
omegau_test = {}

for i in list(set(testplay.user_index.values)):
    omegau_test[i] = list(test_usergroup.get_group(i)['song_index'])
    
trainplay = trainsub[trainsub.plays>0]    
train_usergroup = trainplay.groupby('user_index')
omegau_train = {}

for i in list(set(trainplay.user_index.values)):
    omegau_train[i] = list(train_usergroup.get_group(i)['song_index'])

trainplay = trainsub[trainsub.plays>0] 
train_songgroup = trainplay.groupby('song_index')
omegav_train = {}

for i in list(set(trainplay.song_index.values)):
    omegav_train[i] = list(train_songgroup.get_group(i)['user_index'])   


testpivot = testsub.pivot(index='user_id',columns='sid', values='plays')
M_test = testpivot.as_matrix()
M_test = np.nan_to_num(M_test)







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



def item_based_rec(user_to_song, song_to_user, N2, user_to_song_test):
    
    # Item-based similarity
    
    sim_songs = np.zeros([N2,N2])
    top_songs = {}
    
    
    alpha = 0.5
    # Gamma causes larger weights to be emphasized while dropping smaller weights to zero
    gamma = 1
    
    #Calculate similarity scores for all pairs of songs and get most similar songs to every song
    for song1 in song_to_user:
        top_songs[song1] = []
        for song2 in song_to_user:
            user1 = song_to_user[song1]
            user2 = song_to_user[song2]
            sim_songs[song1][song2] = float(len(set(user1).intersection(user2))) / (len(user1)**alpha * len(user2)**(1-alpha))
            sim_songs[song1][song2] = (sim_songs[song1][song2])**gamma
            top_songs[song1].append((song2, sim_songs[song1][song2]))
        top_songs[song1] = sorted(top_songs[song1], key = itemgetter(1), reverse = True)[0:500]
    
    #Find songs most similar to songs user listens to and recommend those songs
    recommend = {}       
    for user in user_to_song:
        songs = user_to_song[user]
        recommend[user] = []
        for song in songs:
            recommend[user] = recommend[user] + top_songs[song]
        testDict = defaultdict(float)
        for key, val in recommend[user]:
            testDict[key] += val
        recommend[user] = testDict.items()
        recommend[user] = sorted(recommend[user], key = itemgetter(1), reverse = True)
        recommend[user] = [x for x in recommend[user] if x[0] not in user_to_song[user]][0:500]                
        recommend[user] = [x[0] for x in recommend[user]]
    
    #Calculate how good of a prediction
    apk_sum = 0

    counter = 0
    for i in user_to_song_test:
        if i in recommend:
            apk_sum += apk(user_to_song_test[i],recommend[i])
            counter += 1
    
    map = apk_sum/counter
    return map
    
   
    

def user_based_rec(user_to_song, song_to_user, N1, user_to_song_test):
    
    sim_users = np.zeros([N1,N1])
    
    alpha = 0.5
    #Gamma causes larger weights to be emphasized while dropping smaller weights to zero
    gamma = 1
    
    #Find users most similar to each user
    for user1 in user_to_song:
        for user2 in user_to_song:
            song1 = user_to_song[user1]
            song2 = user_to_song[user2]
            sim_users[user1][user2] = float(len(set(song1).intersection(song2))) / (len(song1)**alpha * len(song2)**(1-alpha))
            sim_users[user1][user2] = (sim_users[user1][user2])**gamma            
    
    #For each user, find songs of most similar users and recommend those songs      
    recommend = {}
    for user1 in user_to_song:
        recommend[user1] = []
        for song in song_to_user:
            users = song_to_user[song]
            temp = 0
            for user2 in users:
                temp = temp + sim_users[user1][user2]
            recommend[user1].append((song, temp)) 
        recommend[user1] = sorted(recommend[user1], key = itemgetter(1), reverse = True)
        recommend[user1] = [x for x in recommend[user1] if x[0] not in user_to_song[user1]][0:500]        
        recommend[user1] = [x[0] for x in recommend[user1]]
    
    #Calculate how good of a prediction
    apk_sum = 0

    counter = 0
    for i in user_to_song_test:
        if i in recommend:
            apk_sum += apk(user_to_song_test[i],recommend[i])
            counter += 1
    
    map = apk_sum/counter
    return map



#Call item-based recommendation function
item_based_map = item_based_rec(omegau_train, omegav_train, np.shape(M_train)[1], omegau_test)
print item_based_map

#Call user-based recommendation fucntion
user_based_map = user_based_rec(omegau_train, omegav_train, np.shape(M_train)[0], omegau_test)
print user_based_map
